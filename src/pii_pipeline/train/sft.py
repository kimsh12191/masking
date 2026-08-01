"""좌표 학습 SFT 의 순수 부분 — 대화 구성과 손실 마스킹.

학습 자체는 평범한 supervised fine-tuning 이다. 특별한 손실도, 좌표 회귀
헤드도 없다.

    입력   [system] + [타일 이미지] + [user]
    정답   {"findings":[{"text":"강동혁","bbox_2d":[170,112,261,156]}, ...]}
    손실   **정답 토큰만** cross-entropy

모델은 이미 이 형식으로 답한다. 숫자만 틀리니까 그 숫자를 정답으로 놓고
next-token 예측을 돌리는 것이 전부다.

이 모듈에 torch/transformers 를 들이지 않은 이유
------------------------------------------------

여기서 틀리기 쉬운 것은 모델이 아니라 **손실 마스킹**이다. 프롬프트 토큰까지
손실에 넣으면 모델이 자기 지시문을 외우고, 정답 시작 위치를 한 토큰이라도
잘못 잡으면 좌표가 통째로 밀려서 학습된다. 그런데 GPU 없이는 학습을 못 돌려
보므로, 그 계산만 떼어 내 **CPU 에서 테스트할 수 있게** 했다.
``scripts/train_grounding.py`` 가 이 함수들에 실제 프로세서를 물린다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.prompts import (
    SYSTEM_GROUNDING,
    SYSTEM_LOCATE,
    USER_GROUNDING,
    build_locate_user,
)

#: 손실에서 제외할 라벨 값 (PyTorch cross-entropy 의 기본 ``ignore_index``).
IGNORE_INDEX = -100

#: 과제 두 종류.
#:
#: ==========  ==========================================================
#: ``locate``  **주 과제.** 값을 주고 위치만 묻는다. 손실이 거의 전부 좌표에
#:             걸리고, OCR 오독이 정답에 섞이지 않는다
#: ``read``    전부 읽고 위치까지. 도장·손글씨처럼 **지목할 텍스트가 없는**
#:             영역은 이쪽으로만 가르칠 수 있다
#: ==========  ==========================================================
TASKS = ("locate", "read")

#: **토큰 축을 갖는** 입력 키. 배치에서 길이를 맞춰 패딩해야 하는 것들이다.
#:
#: 나머지(``pixel_values``, ``image_grid_thw``)는 토큰 축이 없다 — Qwen-VL 의
#: 프로세서는 패치를 배치 차원 없이 ``(패치수, 차원)`` 으로 납작하게 돌려주고
#: ``image_grid_thw`` 가 어디부터 어디까지가 몇 번째 이미지인지 알려준다.
#: 그래서 이 둘을 구분하지 않고 일괄로 ``[0]`` 을 떼면 **패치 하나만 남는다.**
#: 증상은 학습이 도는데 이미지를 거의 못 보는 것이고, 로그로는 안 보인다.
TOKEN_AXIS_KEYS: frozenset[str] = frozenset(
    {"input_ids", "attention_mask", "token_type_ids", "labels"}
)


def pad_fill_value(key: str, pad_id: int) -> int:
    """패딩 자리에 넣을 값.

    ``labels`` 만 ``IGNORE_INDEX`` 다. 여기에 ``pad_id`` 를 넣으면 **모델이
    문장 끝에 패딩 토큰을 뱉도록 배운다** — 배치에 길이가 섞여 있을 때만
    나타나서 재현이 까다롭다.

    Raises:
        ValueError: 토큰 축이 없는 키를 넘겼을 때. 그런 텐서는 패딩이 아니라
            이어 붙이는 대상이라 조용히 처리하면 안 된다.
    """
    if key not in TOKEN_AXIS_KEYS:
        raise ValueError(
            f"'{key}' 는 토큰 축이 없습니다. 패딩이 아니라 연결(cat) 대상입니다."
        )
    return IGNORE_INDEX if key == "labels" else (0 if key != "input_ids" else pad_id)


@dataclass
class Example:
    """학습 샘플 하나. ``build_grounding_data.py`` 가 만든 jsonl 한 줄."""

    image_path: Path
    target: dict[str, Any]
    #: 재현·디버깅용. 학습에는 쓰지 않는다.
    meta: dict[str, Any]
    #: ``TASKS`` 중 하나.
    task: str = "locate"
    #: ``locate`` 과제에서 위치를 물을 값들 (중복 없음).
    query: list[str] = field(default_factory=list)

    @property
    def answer(self) -> str:
        """모델이 내야 할 문자열.

        ``separators`` 를 고정해 공백이 흔들리지 않게 한다. 같은 내용이 어떤
        샘플에서는 ``", "``, 다른 샘플에서는 ``","`` 로 나오면 모델이 그
        차이까지 배우느라 용량을 쓴다.
        """
        return json.dumps(self.target, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path: str | Path) -> list[Example]:
    """``build_grounding_data.py`` 의 산출물을 읽는다.

    Args:
        path: ``data.jsonl`` 경로. 이미지 경로는 이 파일 기준 상대경로다.

    Returns:
        샘플 목록.

    Raises:
        FileNotFoundError: jsonl 이 없거나 이미지가 없을 때. **조용히 건너뛰지
            않는다** — 데이터가 반쯤 빠진 채로 학습이 도는 것이 가장 찾기 어렵다.
    """
    jsonl = Path(path)
    root = jsonl.parent
    out: list[Example] = []
    with jsonl.open(encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image = root / row["image"]
            if not image.is_file():
                raise FileNotFoundError(f"{jsonl}:{lineno} 의 이미지가 없습니다: {image}")
            task = row.get("task", "locate")
            if task not in TASKS:
                raise ValueError(
                    f"{jsonl}:{lineno} 의 task 가 잘못되었습니다: {task!r} "
                    f"({' / '.join(TASKS)} 중 하나)"
                )
            if task == "locate" and not row.get("query"):
                # 질문이 비면 "아무것도 안 물었는데 답하라" 가 되어, 모델이
                # 질문에 없는 것을 지어내도록 배운다.
                raise ValueError(f"{jsonl}:{lineno} 의 locate 샘플에 query 가 없습니다")
            out.append(
                Example(
                    image_path=image,
                    target=row["target"],
                    meta=row.get("meta", {}),
                    task=task,
                    query=row.get("query", []),
                )
            )
    return out


def build_messages(example: Example) -> tuple[list[dict[str, Any]], str]:
    """채팅 메시지와 정답 문자열을 만든다.

    Returns:
        ``(프롬프트 메시지, 정답 문자열)``. 메시지에는 assistant 턴이 없다 —
        프로세서에 ``add_generation_prompt=True`` 로 넣어 프롬프트 길이를 재고,
        그 뒤에 정답을 이어 붙인다. 이래야 정답 시작 위치가 **정의상** 정확하다
        (문자열을 붙여 놓고 나중에 찾으면 토크나이저 경계에서 어긋난다).
    """
    if example.task == "locate":
        system, user = SYSTEM_LOCATE, build_locate_user(example.query)
    else:
        system, user = SYSTEM_GROUNDING, USER_GROUNDING

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user},
            ],
        },
    ]
    return messages, example.answer


def mask_prompt(
    input_ids: list[int], prompt_len: int, ignore_index: int = IGNORE_INDEX
) -> list[int]:
    """프롬프트 구간을 손실에서 제외한 라벨을 만든다.

    ``prompt_len`` 은 **정답 직전까지의 토큰 수**다. 여기가 한 토큰이라도
    어긋나면 모델이 한 칸 밀린 것을 배우고, 그 증상은 "좌표가 이상해졌다" 로만
    보인다. 그래서 이 계산을 별도 함수로 빼서 테스트로 못박는다.

    Args:
        input_ids: 프롬프트 + 정답 전체 토큰.
        prompt_len: 프롬프트 토큰 수.
        ignore_index: 손실 제외 값.

    Returns:
        ``input_ids`` 와 같은 길이의 라벨.

    Raises:
        ValueError: ``prompt_len`` 이 전체 길이를 넘거나 음수일 때. 이 상태로
            학습하면 손실이 0 이 되어 "학습은 도는데 아무것도 안 배운다".
    """
    total = len(input_ids)
    if not 0 <= prompt_len <= total:
        raise ValueError(
            f"prompt_len={prompt_len} 이 전체 길이 {total} 범위를 벗어났습니다. "
            "프롬프트 렌더링과 토크나이즈가 어긋났을 때 발생합니다."
        )
    if prompt_len == total:
        raise ValueError(
            "정답 토큰이 하나도 없습니다 (prompt_len == 전체 길이). "
            "손실이 0 이 되어 학습이 도는데 아무것도 배우지 않습니다."
        )
    return [ignore_index] * prompt_len + list(input_ids[prompt_len:])


def summarize(examples: list[Example]) -> dict[str, Any]:
    """학습 전 데이터 점검용 통계. **돌리기 전에 눈으로 볼 것.**"""
    n_items = sum(len(e.target.get("findings", [])) for e in examples)
    n_textless = sum(
        1
        for e in examples
        for f in e.target.get("findings", [])
        if not f.get("text")
    )
    lengths = [len(e.answer) for e in examples]
    by_task: dict[str, int] = {}
    for e in examples:
        by_task[e.task] = by_task.get(e.task, 0) + 1
    return {
        "samples": len(examples),
        "by_task": by_task,
        "items": n_items,
        "items_per_sample": round(n_items / max(1, len(examples)), 1),
        "textless_items": n_textless,
        "answer_chars_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "answer_chars_max": max(lengths, default=0),
    }
