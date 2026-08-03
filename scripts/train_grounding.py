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

    LLM             LoRA
    vision merger   전체 학습        (--no-merger 로 끔)
    ViT             동결             (--vision-blocks N 으로 상위 N개 블록 열기)

ViT 를 기본으로 열지 않는 이유는 **순서** 때문이다. merger 만 열고 먼저
재야 어디까지가 merger 몫인지 안다. 한꺼번에 열면 좋아져도 나빠져도 원인을
못 가른다. merger 만으로 격자 아래로 못 내려가면 그때 ``--vision-blocks`` 를 켠다.

모듈 이름은 모델·라이브러리 버전에 따라 다르므로 ``--list-modules`` 로
확인하고 ``--merger-module`` / ``--vision-prefix`` 로 지정할 수 있게 해 두었다.
ViT LoRA 대상은 하드코딩하지 않고 모델에서 찾는다 (``select_vision_blocks``) —
LLM 이 ``q_proj``/``k_proj`` 인 반면 ViT 는 ``qkv`` 로 합쳐져 있고 그마저도
버전마다 다르다.

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
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pii_pipeline.config import load_config  # noqa: E402
from pii_pipeline.train.config import TrainConfig  # noqa: E402
from pii_pipeline.train.sft import (  # noqa: E402
    IGNORE_INDEX,
    TOKEN_AXIS_KEYS,
    Example,
    build_messages,
    describe_sizes,
    expected_tile_size,
    load_jsonl,
    mask_prompt,
    pad_fill_value,
    resolve_image_size,
    select_vision_blocks,
    summarize,
    tally_image_sizes,
)

log = logging.getLogger("train_grounding")



# --------------------------------------------------------------------------
# 데이터셋 — 라이브러리 버전에 민감한 부분은 전부 여기 모아 둔다
# --------------------------------------------------------------------------


def resize_to(image: Any, size: tuple[int, int]) -> Any:
    """PIL 이미지를 ``(폭, 높이)`` 로 맞춘다.

    **좌표는 손대지 않는다.** 정답이 per-mille(0~1000 상대좌표)이라 이미지
    크기에 불변이다. 이 성질이 없으면 강제 리사이즈는 라벨을 망가뜨리는 짓이
    되므로, 여기서 좌표를 건드리지 않는 것이 우연이 아니라 전제다.

    보간은 ``build_grounding_data.to_tile_size`` 의 방침을 따른다 — 줄일 때는
    면적 평균에 가까운 LANCZOS, 늘릴 때는 BICUBIC. 작은 한글에서 이 선택이
    무의미하지 않다.

    이 함수를 타는 샘플은 **애초에 크기가 틀린 것들뿐이다.** 맞는 샘플은
    호출되지 않으므로 보간이 한 번도 들어가지 않는다. 즉 정상 데이터의 픽셀은
    추론 타일과 그대로 같고, 틀린 데이터만 보간 비용을 낸다.
    """
    from PIL import Image  # type: ignore[import-not-found]

    shrinking = size[0] * size[1] < image.size[0] * image.size[1]
    return image.resize(size, Image.LANCZOS if shrinking else Image.BICUBIC)


class GroundingDataset:
    """``Example`` 을 모델 입력으로 바꾼다.

    **프롬프트 길이를 재는 방법이 이 클래스의 요점이다.** 정답이 시작하는
    토큰 위치를 문자열에서 찾으면 토크나이저 경계에서 어긋난다. 대신
    프롬프트만 따로 한 번 인코딩해 그 길이를 쓴다 — 같은 이미지를 넣으므로
    비전 토큰 확장 개수도 같고, 따라서 길이가 **정의상** 정확하다.

    이미지를 두 번 전처리하는 비용이 들지만, 여기서 틀리면 학습 전체가
    조용히 무의미해진다. 정확도 쪽에 값을 지불한다.
    """

    def __init__(
        self,
        examples: list[Example],
        processor: Any,
        max_len: int,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.examples = examples
        self.processor = processor
        self.max_len = max_len
        #: 모든 입력을 강제로 맞출 ``(폭, 높이)``. ``None`` 이면 손대지 않는다.
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image  # type: ignore[import-not-found]

        example = self.examples[index]
        messages, answer = build_messages(example)
        image = Image.open(example.image_path).convert("RGB")
        if self.image_size is not None and image.size != self.image_size:
            image = resize_to(image, self.image_size)

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
        rows = [item[key] for item in batch]
        # 이어 붙이는 축은 0 번뿐이고 **나머지 축은 전부 같아야 한다.** 다르면
        # torch 가 "Sizes of tensors must match except in dimension 0" 으로
        # 죽는데, 그 메시지에는 어느 키인지도 왜인지도 없다. DataLoader 워커
        # 안에서 터져 traceback 이 두 겹이라 더 알아보기 어렵다.
        #
        # **여기는 백스톱이다.** ``GroundingDataset`` 이 모든 이미지를 추론 크기로
        # 맞추므로 정상 경로에서는 걸리지 않는다. 그래도 남겨 두는 이유는, 걸린다면
        # 그건 크기 맞추기가 새는 것이라는 뜻이고 그 사실이 메시지에 있어야 한다.
        tail = rows[0].shape[1:]
        for i, row in enumerate(rows[1:], 1):
            if row.shape[1:] != tail:
                raise ValueError(
                    f"'{key}' 의 모양이 샘플마다 다릅니다: "
                    f"{tuple(rows[0].shape)} vs {tuple(row.shape)} (샘플 {i}).\n"
                    "비전 텐서는 패딩이 아니라 이어 붙이는 대상이라 첫 축 말고는 "
                    "모두 같아야 합니다.\n"
                    "입력 크기 맞추기가 동작하지 않았습니다 — GroundingDataset 이 "
                    "image_size 를 받았는지 확인하세요 (시작 로그에 맞출 크기가 찍힙니다)."
                )
        out[key] = torch.cat(rows, dim=0)
    return out


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------


def load_model(model_id: str, dtype: str, device_map: str | None = None) -> tuple[Any, Any]:
    """모델과 프로세서를 올린다.

    VLM 오토클래스 이름이 transformers 버전마다 다르다 (``AutoModelForImageTextToText``
    가 새 이름, ``AutoModelForVision2Seq`` 가 옛 이름). 둘 다 시도하고, 없으면
    무엇을 올려야 하는지 알려준다 — "왜 안 되지" 로 시간을 쓰지 않게.

    Args:
        device_map: ``--list-modules`` 처럼 추론만 할 때 ``"auto"``. **학습에서는
            ``None``** 이다. Trainer 가 모델 배치를 직접 관리하는데 ``device_map``
            으로 이미 쪼개 놓으면 충돌한다 (9B bf16 은 80GB 한 장에 들어간다).
    """
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]

    auto_vlm = None
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        auto_vlm = getattr(transformers, name, None)
        if auto_vlm is not None:
            log.info("오토클래스: transformers.%s", name)
            break
    if auto_vlm is None:  # pragma: no cover - 환경 의존
        raise RuntimeError(
            "transformers 에 VLM 오토클래스가 없습니다 "
            "(AutoModelForImageTextToText / AutoModelForVision2Seq). "
            "Qwen3-VL 을 지원하는 버전으로 올리세요."
        )

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    processor = transformers.AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = auto_vlm.from_pretrained(
        model_id, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map
    )
    # gradient checkpointing 과 충돌한다. PEFT 로 감싸기 전에 꺼 둔다.
    model.config.use_cache = False
    return model, processor


def attach_lora(model: Any, tc: TrainConfig) -> Any:
    """LoRA 를 걸고, merger 는 전체 학습 대상으로 올린다.

    세 부위를 따로 다룬다.

    ==============  ==================================================
    LLM             LoRA (이름이 표준적이라 접미사로 지정)
    vision merger   **전체 학습.** 32px 토큰 격자 아래의 위치 정보가
                    여기서 살아남느냐로 결정된다
    ViT 상위 블록   LoRA. 이름을 모델에서 찾아 전체 경로로 지정한다
                    (``select_vision_blocks``)
    ==============  ==================================================
    """
    import torch.nn as nn  # type: ignore[import-not-found]
    from peft import LoraConfig, get_peft_model  # type: ignore[import-not-found]

    targets = list(tc.lora_targets)
    merger = tc.merger_module if tc.train_merger else None

    if tc.vision_blocks > 0:
        linear_names = [
            name for name, module in model.named_modules() if isinstance(module, nn.Linear)
        ]
        picked = select_vision_blocks(linear_names, tc.vision_prefix, tc.vision_blocks)
        if not picked:
            raise ValueError(
                f"비전 타워 '{tc.vision_prefix}' 에서 블록 선형층을 찾지 못했습니다. "
                f"--list-modules 로 실제 접두사를 확인하고 --vision-prefix 로 "
                f"지정하거나, --vision-blocks 0 으로 끄세요."
            )
        targets += picked
        blocks = sorted({n.rsplit(".blocks.", 1)[1].split(".")[0] for n in picked})
        log.info("ViT LoRA: 블록 %s (선형층 %d개)", ",".join(blocks), len(picked))

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
        r=tc.rank,
        lora_alpha=tc.alpha,
        lora_dropout=tc.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        modules_to_save=modules_to_save,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


# --------------------------------------------------------------------------
# dry-run — GPU 없이 볼 수 있는 것까지 본다
# --------------------------------------------------------------------------


def dry_run(
    examples: list[Example],
    model_id: str,
    index: int,
    image_size: tuple[int, int] | None = None,
) -> int:
    """데이터와 프롬프트를 확인한다. 가능하면 손실 마스킹까지.

    Args:
        image_size: 학습이 강제로 맞출 크기. **여기서도 맞춰야 한다** — 아래에
            찍는 토큰 수가 비전 토큰을 포함하므로, 맞추지 않으면 실제 학습과
            다른 숫자를 보고하고 max_len 판단이 어긋난다.
    """
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
    if image_size is not None and image.size != image_size:
        print(f"\n(이미지 {image.size[0]}x{image.size[1]} -> {image_size[0]}x{image_size[1]} 로 맞춤)")
        image = resize_to(image, image_size)
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
    ap.add_argument("--config", "-c", default=None, help="YAML 설정 (서빙과 같은 것)")
    ap.add_argument("--model", default=None, help="베이스 모델 (서빙과 같아야 한다)")
    # 기본값은 전부 설정 파일(train 섹션)에서 온다. 여기 default 가 None 인 것은
    # **준 것만 덮기** 위해서다 — argparse 기본값을 두면 설정 파일을 항상 무시한다.
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--alpha", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--merger-module", default=None)
    ap.add_argument(
        "--vision-blocks",
        type=int,
        default=None,
        help="비전 인코더(ViT)의 **상위 N개 블록**에 LoRA 를 건다. 0 이면 ViT 동결. "
        "merger 만으로 32px 격자 아래로 못 내려가면 여기를 연다 — 다만 "
        "**merger 만 열고 먼저 재 볼 것.** 한꺼번에 열면 뭐가 들었는지 모른다",
    )
    ap.add_argument(
        "--vision-prefix",
        default=None,
        help="비전 타워 모듈 접두사. --list-modules 로 확인",
    )
    ap.add_argument(
        "--no-merger",
        action="store_true",
        help="merger 를 동결한다. 32px 토큰 격자 아래로는 못 내려간다 — A/B 비교용",
    )
    ap.add_argument("--merge", default=None, help="학습 후 머지해 저장할 경로 (서빙용)")
    ap.add_argument("--workers", type=int, default=None, help="데이터로더 워커 (이미지 전처리)")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="앞에서 N개만 사용. **처음에는 8 정도로 한 번 돌려볼 것** — "
        "배선이 틀렸는지 몇 분 만에 안다",
    )
    ap.add_argument("--dry-run", action="store_true", help="데이터·프롬프트·마스킹만 확인")
    ap.add_argument("--dry-run-index", type=int, default=0)
    ap.add_argument("--list-modules", action="store_true", help="모듈 이름 출력 후 종료")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 설정 파일(train 섹션)이 기본, CLI 가 덮는다 (준 것만).
    # 전체 설정을 들고 있어야 한다 — 학습 입력 크기를 **추론 쪽 설정**
    # (pipeline.canvas / detect.tiles / llm.image_max_side) 에서 끌어내기 때문이다.
    cfg = load_config(args.config)
    tc = cfg.train
    for name in (
        "model", "epochs", "lr", "batch", "grad_accum", "rank", "alpha", "dropout",
        "max_len", "dtype", "merger_module", "vision_blocks", "vision_prefix",
        "workers", "seed",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(tc, name, value)
    if args.no_merger:
        tc.train_merger = False

    if args.list_modules:
        model, _ = load_model(tc.model, tc.dtype, device_map="auto")
        for name, _ in model.named_modules():
            print(name)
        return 0

    examples = load_jsonl(args.data)
    if args.max_samples:
        examples = examples[: args.max_samples]
    if not examples:
        print("학습 샘플이 없습니다.", file=sys.stderr)
        return 1

    # 입력 크기를 하나로 고정한다. **추론에서 오는 크기가 기준이다** — 캔버스가
    # 고정이라 추론 타일은 항상 한 가지 크기이고, 학습이 거기에 맞아야 모델이
    # 상대할 기하가 하나로 유지된다.
    #
    # 모델을 GPU 에 올리기 전에 재는 이유: 크기가 섞여 있으면 예전에는 첫 에폭
    # 중간에 collate 가 죽었는데, 그때까지 9B 를 올리고 기다린 시간이 전부
    # 낭비였다. 지금은 죽지 않고 맞추지만, 무엇을 맞췄는지는 시작할 때 보여야 한다.
    pipe = cfg.pipeline
    expected = expected_tile_size(
        pipe.canvas,
        pipe.detect.tiles,
        pipe.detect.overlap,
        pipe.detect.image_factor,
        pipe.llm.image_max_side,
    )
    sizes = tally_image_sizes(examples)
    image_size = resolve_image_size(sizes, expected)
    n_off = sum(n for wh, n in sizes.items() if wh != image_size)

    if expected is None:
        log.warning(
            "pipeline.canvas 가 고정이 아니라 추론 입력 크기도 하나가 아닙니다. "
            "데이터의 최다 크기(%dx%d)로 맞춥니다 — 임의의 선택입니다. "
            "좌표 학습에는 canvas 를 고정하세요 (config/default.yaml 주석 참조).",
            *image_size,
        )
    if n_off:
        log.warning(
            "학습 이미지 %d장이 추론 크기(%dx%d)와 달라 로드할 때 맞춥니다. "
            "좌표는 per-mille 이라 그대로 유효합니다. 다만 보간이 한 번 더 들어가므로, "
            "데이터를 그 크기로 다시 만드는 것이 낫습니다.\n분포:\n%s",
            n_off,
            *image_size,
            describe_sizes(sizes),
        )
    else:
        log.info("학습 이미지 %d장, 모두 %dx%d (추론과 같다)", len(examples), *image_size)

    if args.dry_run:
        return dry_run(examples, tc.model, args.dry_run_index, image_size)

    from transformers import Trainer, TrainingArguments  # type: ignore[import-not-found]

    model, processor = load_model(tc.model, tc.dtype)
    model = attach_lora(model, tc)

    # gradient checkpointing + LoRA 의 고전적인 함정. 베이스가 전부 동결이라
    # 체크포인트 구간 입력에 grad_fn 이 없고, 역전파가
    # "element 0 of tensors does not require grad" 로 죽는다. 임베딩 출력에
    # requires_grad 를 세워 사슬을 잇는다.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = GroundingDataset(examples, processor, tc.max_len, image_size)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError(
            "토크나이저에 pad_token_id 도 eos_token_id 도 없습니다. "
            "패딩 값을 정할 수 없어 배치를 만들 수 없습니다."
        )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=tc.epochs,
            learning_rate=tc.lr,
            per_device_train_batch_size=tc.batch,
            gradient_accumulation_steps=tc.grad_accum,
            gradient_checkpointing=True,
            # reentrant 방식은 PEFT 와 섞였을 때 일부 파라미터의 grad 를 흘린다.
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=tc.dtype == "bf16",
            fp16=tc.dtype == "fp16",
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
            seed=tc.seed,
            dataloader_num_workers=tc.workers,
            # pixel_values 는 모델 시그니처에 없는 이름일 수 있다. 끄지 않으면
            # Trainer 가 조용히 버리고 모델이 이미지를 못 본다.
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, pad_id),
    )
    trainer.train()

    out_dir = Path(args.out)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    (out_dir / "train_args.json").write_text(
        json.dumps({"args": vars(args), "train": dataclasses.asdict(tc)},
                   ensure_ascii=False, indent=2), encoding="utf-8"
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
