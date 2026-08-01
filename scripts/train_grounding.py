#!/usr/bin/env python3
"""Qwen3-VL 좌표 LoRA 학습.

    build_grounding_data.py  →  tiles/*.png + data.jsonl
    train_grounding.py       →  LoRA 어댑터 (+ 머지된 서빙용 가중치)

평범한 SFT 다. 좌표 회귀 헤드도, 특별한 손실도 없다::

    입력   [system] + [타일 이미지] + [user]
    정답   {"findings":[{"text":"강동혁","bbox_2d":[318,266,398,317]}, ...]}
    손실   정답 토큰만 cross-entropy

모델은 이미 이 형식으로 답한다. **숫자만 틀리니까 그 숫자를 정답으로 놓고
next-token 예측을 돌리는 것이 전부다.**

무엇을 업데이트하는가
---------------------

LLM 만 열면 부족하다. 지금 오차가 비전 토큰 1칸(32px) 규모인데, 토큰 격자
아래의 위치 정보는 **vision merger** 가 4개 패치를 1개 토큰으로 압축하는
지점에서 결정된다. 거기를 닫아 두면 격자 아래로 못 내려간다. 그래서
LLM 은 LoRA, merger 는 전체 학습이 기본값이다.

모듈 이름은 모델·라이브러리 버전에 따라 다르므로 ``--list-modules`` 로
확인하고 ``--merger-module`` 로 지정할 수 있게 해 두었다.

돌리기 전에
-----------

``--dry-run`` 으로 **프롬프트 렌더링과 손실 마스킹을 먼저 눈으로 볼 것.**
프롬프트가 손실에 섞이거나 정답 시작이 한 토큰 밀리면 학습 로그에는 아무
징후가 없고 "좌표가 안 좋아지네" 로만 나타난다.

사용법::

    python scripts/train_grounding.py --data data/grounding/data.jsonl --dry-run
    python scripts/train_grounding.py --data data/grounding/data.jsonl -o out/lora
    python scripts/train_grounding.py --data ... -o out/lora --merge out/merged
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pii_pipeline.train.sft import (  # noqa: E402
    IGNORE_INDEX,
    TOKEN_AXIS_KEYS,
    Example,
    build_messages,
    load_jsonl,
    mask_prompt,
    pad_fill_value,
    summarize,
)

log = logging.getLogger("train_grounding")

#: LoRA 를 걸 선형층. Qwen 계열 트랜스포머 블록의 표준 이름이다.
DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

#: 전체 학습할 비전 merger 모듈. **32px 토큰 격자 아래의 정밀도가 여기 걸린다.**
#: 모델 구현에 따라 이름이 다르므로 ``--list-modules`` 로 확인할 것.
DEFAULT_MERGER = "visual.merger"


# --------------------------------------------------------------------------
# 데이터셋 — 라이브러리 버전에 민감한 부분은 전부 여기 모아 둔다
# --------------------------------------------------------------------------


class GroundingDataset:
    """``Example`` 을 모델 입력으로 바꾼다.

    **프롬프트 길이를 재는 방법이 이 클래스의 요점이다.** 정답이 시작하는
    토큰 위치를 문자열에서 찾으면 토크나이저 경계에서 어긋난다. 대신
    프롬프트만 따로 한 번 인코딩해 그 길이를 쓴다 — 같은 이미지를 넣으므로
    비전 토큰 확장 개수도 같고, 따라서 길이가 **정의상** 정확하다.

    이미지를 두 번 전처리하는 비용이 들지만, 여기서 틀리면 학습 전체가
    조용히 무의미해진다. 정확도 쪽에 값을 지불한다.
    """

    def __init__(self, examples: list[Example], processor: Any, max_len: int) -> None:
        self.examples = examples
        self.processor = processor
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image  # type: ignore[import-not-found]

        example = self.examples[index]
        messages, answer = build_messages(example)
        image = Image.open(example.image_path).convert("RGB")

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        eos = self.processor.tokenizer.eos_token or ""
        full_text = prompt_text + answer + eos

        prompt = self.processor(text=[prompt_text], images=[image], return_tensors="pt")
        full = self.processor(text=[full_text], images=[image], return_tensors="pt")

        import torch  # type: ignore[import-not-found]

        ids = full["input_ids"][0].tolist()
        prompt_len = int(prompt["input_ids"].shape[1])
        labels = mask_prompt(ids, prompt_len)

        # 토큰 축이 있는 것만 배치 차원을 뗀다. pixel_values / image_grid_thw 는
        # 애초에 배치 차원이 없으므로 [0] 을 떼면 패치 하나만 남는다
        # (TOKEN_AXIS_KEYS 참조).
        item = {
            key: (value[0] if key in TOKEN_AXIS_KEYS else value)
            for key, value in full.items()
        }
        item["labels"] = torch.tensor(labels, dtype=torch.long)

        if len(ids) > self.max_len:
            log.warning(
                "샘플 %d 이 max_len(%d)을 넘어 잘렸습니다 (%d 토큰). "
                "뒤쪽 항목의 좌표를 못 배웁니다 — 타일을 더 쪼개거나 max_len 을 올리세요.",
                index,
                self.max_len,
                len(ids),
            )
            for key in TOKEN_AXIS_KEYS & item.keys():
                item[key] = item[key][: self.max_len]
        return item


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    """가변 길이를 오른쪽 패딩으로 맞춘다.

    토큰 축이 있는 것만 패딩하고, ``pixel_values`` / ``image_grid_thw`` 처럼
    없는 것은 이어 붙인다 (``TOKEN_AXIS_KEYS`` 참조).
    """
    import torch  # type: ignore[import-not-found]

    longest = max(item["input_ids"].shape[0] for item in batch)
    out: dict[str, Any] = {}

    for key in TOKEN_AXIS_KEYS & batch[0].keys():
        fill = pad_fill_value(key, pad_id)
        rows = []
        for item in batch:
            row = item[key]
            gap = longest - row.shape[0]
            if gap > 0:
                row = torch.cat([row, torch.full((gap,), fill, dtype=row.dtype)])
            rows.append(row)
        out[key] = torch.stack(rows)

    for key in batch[0].keys() - out.keys():
        out[key] = torch.cat([item[key] for item in batch], dim=0)
    return out


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------


def load_model(model_id: str, dtype: str) -> tuple[Any, Any]:
    """모델과 프로세서를 올린다.

    ``AutoModelForVision2Seq`` 가 없는 구버전 transformers 를 위해 예외 메시지를
    구체적으로 남긴다 — "왜 안 되지" 로 시간을 쓰지 않게.
    """
    import torch  # type: ignore[import-not-found]
    from transformers import AutoProcessor  # type: ignore[import-not-found]

    try:
        from transformers import AutoModelForVision2Seq as AutoVlm  # type: ignore
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise RuntimeError(
            "transformers 에 AutoModelForVision2Seq 가 없습니다. "
            "Qwen3-VL 을 지원하는 버전으로 올리세요."
        ) from exc

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoVlm.from_pretrained(
        model_id, torch_dtype=torch_dtype, trust_remote_code=True, device_map="auto"
    )
    return model, processor


def attach_lora(
    model: Any, rank: int, alpha: int, dropout: float, merger: str | None
) -> Any:
    """LoRA 를 걸고, merger 는 전체 학습 대상으로 올린다."""
    from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]

    modules_to_save = None
    if merger:
        found = [n for n, _ in model.named_modules() if merger in n]
        if not found:
            raise ValueError(
                f"merger 모듈 '{merger}' 를 찾지 못했습니다. "
                f"--list-modules 로 실제 이름을 확인하거나 --no-merger 로 끄세요."
            )
        modules_to_save = [merger]
        log.info("merger 전체 학습: %s (하위 모듈 %d개)", merger, len(found))

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_LORA_TARGETS,
        modules_to_save=modules_to_save,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


# --------------------------------------------------------------------------
# dry-run — GPU 없이 볼 수 있는 것까지 본다
# --------------------------------------------------------------------------


def dry_run(examples: list[Example], model_id: str, index: int) -> int:
    """데이터와 프롬프트를 확인한다. 가능하면 손실 마스킹까지."""
    print("=" * 68)
    print(" 데이터 요약")
    print("=" * 68)
    for key, value in summarize(examples).items():
        print(f"  {key:24} {value}")

    example = examples[index]
    messages, answer = build_messages(example)
    print()
    print("=" * 68)
    print(f" 샘플 {index}  ({example.image_path.name})")
    print("=" * 68)
    for message in messages:
        for part in message["content"]:
            body = part.get("text", "<이미지>")
            print(f"\n[{message['role']}] {body}")
    print(f"\n[정답 — 이 토큰에만 손실] {answer[:400]}{'...' if len(answer) > 400 else ''}")

    try:
        from transformers import AutoProcessor  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        print("\n(transformers/Pillow 없음 — 토큰 단위 검증은 건너뜁니다)")
        return 0

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - 모델 미반입 환경
        print(f"\n(프로세서를 못 불러옴: {type(exc).__name__} — 토큰 검증 건너뜁니다)")
        return 0

    image = Image.open(example.image_path).convert("RGB")
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    eos = processor.tokenizer.eos_token or ""
    prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")
    full = processor(text=[prompt_text + answer + eos], images=[image], return_tensors="pt")

    ids = full["input_ids"][0].tolist()
    prompt_len = int(prompt["input_ids"].shape[1])
    labels = mask_prompt(ids, prompt_len)
    supervised = [i for i in labels if i != IGNORE_INDEX]

    print()
    print("=" * 68)
    print(" 손실 마스킹 검증")
    print("=" * 68)
    print(f"  전체 토큰      {len(ids)}")
    print(f"  프롬프트(마스킹) {prompt_len}")
    print(f"  정답(손실 대상)  {len(supervised)}")
    print(f"\n  손실이 걸리는 문자열 (디코딩):")
    print(f"    {processor.tokenizer.decode(supervised)[:400]}")
    print("\n  ↑ **정답 JSON 만** 나와야 합니다. 지시문이 섞여 있으면 마스킹이 틀린 것입니다.")
    return 0


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen3-VL 좌표 LoRA 학습")
    ap.add_argument("--data", required=True, help="build_grounding_data.py 의 data.jsonl")
    ap.add_argument("-o", "--out", default="out/lora", help="어댑터 저장 경로")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="베이스 모델 (서빙과 같아야 한다)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--merger-module", default=DEFAULT_MERGER)
    ap.add_argument(
        "--no-merger",
        action="store_true",
        help="merger 를 동결한다. 32px 토큰 격자 아래로는 못 내려간다 — A/B 비교용",
    )
    ap.add_argument("--merge", default=None, help="학습 후 머지해 저장할 경로 (서빙용)")
    ap.add_argument("--dry-run", action="store_true", help="데이터·프롬프트·마스킹만 확인")
    ap.add_argument("--dry-run-index", type=int, default=0)
    ap.add_argument("--list-modules", action="store_true", help="모듈 이름 출력 후 종료")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list_modules:
        model, _ = load_model(args.model, args.dtype)
        for name, _ in model.named_modules():
            print(name)
        return 0

    examples = load_jsonl(args.data)
    if not examples:
        print("학습 샘플이 없습니다.", file=sys.stderr)
        return 1

    if args.dry_run:
        return dry_run(examples, args.model, args.dry_run_index)

    from transformers import Trainer, TrainingArguments  # type: ignore[import-not-found]

    model, processor = load_model(args.model, args.dtype)
    model = attach_lora(
        model,
        args.rank,
        args.alpha,
        args.dropout,
        None if args.no_merger else args.merger_module,
    )
    model.config.use_cache = False  # gradient checkpointing 과 충돌한다

    dataset = GroundingDataset(examples, processor, args.max_len)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,
            bf16=args.dtype == "bf16",
            fp16=args.dtype == "fp16",
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
            seed=args.seed,
            remove_unused_columns=False,  # pixel_values 를 Trainer 가 버리지 않게
        ),
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, pad_id),
    )
    trainer.train()

    out_dir = Path(args.out)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    (out_dir / "train_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n어댑터 저장: {out_dir}")

    if args.merge:
        # vLLM 의 LoRA 핫스왑은 비전타워 쪽 보장이 애매하다. 머지한 가중치로
        # 올리면 그 문제가 없고, 어차피 모델 하나만 쓴다.
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merge)
        processor.save_pretrained(args.merge)
        print(f"머지 저장 (서빙용): {args.merge}")

    print("\n다음 — 반드시 **두 축을 함께** 재라:")
    print("  좌표   python scripts/diagnose.py <평가 페이지들>")
    print("  재현율 python scripts/measure.py <평가 페이지들>")
    print("  재현율이 떨어졌으면 좌표가 좋아졌어도 실패다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
