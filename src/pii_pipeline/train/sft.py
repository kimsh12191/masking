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
from collections import Counter
from collections.abc import Callable
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
#: ``locate``  **기본이자 사실상 유일한 과제.** 값을 주고 위치만 묻는다.
#:             손실이 거의 전부 좌표에 걸리고, OCR 오독이 정답에 섞이지 않는다
#: ``read``    전부 읽고 위치까지. 도장·손글씨처럼 지목할 텍스트가 없는 영역을
#:             가르칠 수 있지만 **기본으로 만들지 않는다** — 좌표 능력은 내용과
#:             무관해서 글자로 배운 것이 도장에도 쓰이고, 과제를 둘로 늘리면
#:             "가끔 text 를 빈 문자열로 낸다" 까지 배운다.
#:             ``build_grounding_data.py --read-ratio`` 로 켠다
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


def select_vision_blocks(
    linear_names: list[str], prefix: str, last_n: int
) -> list[str]:
    """비전 타워의 **상위 ``last_n`` 개 블록** 안에 있는 선형층 이름을 고른다.

    LoRA 대상 이름을 하드코딩할 수 없어서 필요한 함수다. Qwen 계열은 LLM 이
    ``q_proj``/``k_proj``... 인데 ViT 는 ``qkv`` 로 합쳐져 있고, 그마저도 버전마다
    다르다 (2.5-VL 은 MLP 가 ``fc1``/``fc2``, 그 뒤로는 SwiGLU 로 바뀌었다).
    그래서 이름을 짐작하지 않고 **모델에서 찾아** 전체 경로로 지정한다.

    상위 블록만 여는 이유는 하위 블록이 선·획 같은 저수준 특징을 담당해서
    좌표 과제로 흔들 이유가 없고, 흔들면 전사 능력까지 함께 흔들리기 때문이다.

    블록 번호가 없는 모듈(``patch_embed``, ``merger``)은 제외한다 — merger 는
    LoRA 가 아니라 전체 학습으로 따로 다룬다.

    Args:
        linear_names: 모델의 선형층 전체 이름 목록.
        prefix: 비전 타워 접두사 (Qwen 은 ``"visual"``).
        last_n: 뒤에서 몇 개 블록을 열지. 0 이면 열지 않는다.

    Returns:
        LoRA 를 걸 모듈의 **전체 경로** 목록. 접미사가 아니라 전체 경로라야
        같은 이름의 다른 블록이 딸려오지 않는다.
    """
    if last_n <= 0:
        return []

    import re

    block_re = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    by_block: dict[int, list[str]] = {}
    for name in linear_names:
        if not name.startswith(prefix):
            continue
        match = block_re.search(name)
        if match is None:
            continue
        by_block.setdefault(int(match.group(1)), []).append(name)

    if not by_block:
        return []
    keep = sorted(by_block)[-last_n:]
    return [name for index in keep for name in by_block[index]]


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


# --------------------------------------------------------------------------
# 입력 크기 균일성 검사
# --------------------------------------------------------------------------


def image_size(path: Path) -> tuple[int, int]:
    """이미지의 ``(폭, 높이)``.

    **헤더만 읽는다** (``PIL.Image.open`` 은 지연 로딩이다). 픽셀을 디코드하지
    않으므로 수천 장을 재도 싸고, 그래서 학습 시작 전에 전수 검사할 수 있다.
    """
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(path) as im:
        return im.size


def tally_image_sizes(
    examples: list[Example],
    size_of: Callable[[Path], tuple[int, int]] | None = None,
) -> Counter[tuple[int, int]]:
    """학습 이미지의 크기 분포. ``{(폭, 높이): 장수}``.

    Args:
        examples: 학습 샘플.
        size_of: 크기를 읽는 함수. 기본은 ``image_size``.
            **테스트에서 Pillow 없이 돌리기 위한 이음새다.**
    """
    read = size_of or image_size
    return Counter(read(e.image_path) for e in examples)


def describe_sizes(sizes: Counter[tuple[int, int]], limit: int = 8) -> str:
    """크기 분포를 사람이 읽을 줄로. 길면 자른다 (수십 종이면 화면을 덮는다)."""
    lines = [f"  {w}x{h}  {n}장" for (w, h), n in sizes.most_common()[:limit]]
    if len(sizes) > limit:
        lines.append(f"  ... 그 밖에 {len(sizes) - limit}종")
    return "\n".join(lines)


def expected_tile_size(
    canvas: tuple[int, int] | None,
    tiles: int,
    overlap: float,
    image_factor: int,
    image_max_side: int,
) -> tuple[int, int] | None:
    """**추론이 VLM 에 보내는 타일 한 장의 픽셀 크기.** ``(폭, 높이)``.

    학습 입력을 이 크기로 강제하기 위한 것이다. 추론에서는 크기가 항상 이
    하나이므로(캔버스 고정 -> 타일 고정), 학습도 여기에 맞춰야 모델이 상대할
    기하가 하나로 유지된다.

    계산은 ``build_grounding_data.py`` 와 **같은 경로**를 따른다. 그쪽이
    ``tile_rects`` 의 첫 조각 크기를 쓰고 ``as_model_sees`` 로 ``image_max_side``
    축소를 한 번 더 거는데, 여기서도 같은 두 단계를 밟는다. 한쪽만 바뀌면
    빌더가 만든 크기와 학습이 기대하는 크기가 갈리므로 함수를 나란히 두지 않고
    같은 원시 함수(``tile_rects`` / ``fit_max_side``)를 부른다.

    Args:
        canvas: ``pipeline.canvas``. ``None`` 이면 페이지 크기가 입력마다 달라
            고정할 수 있는 크기가 **없다** — 그대로 ``None`` 을 돌려준다.
        tiles: ``detect.tiles``.
        overlap: ``detect.overlap``.
        image_factor: ``detect.image_factor`` (패치 격자).
        image_max_side: ``llm.image_max_side``.

    Returns:
        ``(폭, 높이)``, 또는 캔버스가 고정이 아니면 ``None``.
    """
    if canvas is None:
        return None

    # 지연 임포트 — 이 모듈은 torch 없이 임포트되는 것이 계약이고, 무거운 것을
    # 위로 올리면 그 계약이 조용히 깨진다. 둘 다 순수 함수다.
    from ..detect import tile_rects
    from ..llm.client import fit_max_side

    page_w, page_h = int(canvas[0]), int(canvas[1])
    x1, y1, x2, y2 = tile_rects(page_w, page_h, tiles, overlap, image_factor)[0]
    tile_w = round((x2 - x1) * page_w)
    tile_h = round((y2 - y1) * page_h)
    new_h, new_w = fit_max_side(tile_h, tile_w, image_max_side)
    return new_w, new_h


def resolve_image_size(
    sizes: Counter[tuple[int, int]],
    expected: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """학습 입력을 **강제로 맞출** 크기를 정한다.

    추론 크기(``expected``)가 있으면 무조건 그것이다. 데이터가 다른 크기로
    만들어져 있어도 학습 쪽에서 맞춘다 — 데이터를 다시 만들지 않아도 되고,
    빌더 설정과 학습 설정이 갈라져도 학습이 **추론 기하로 수렴한다.**

    ``expected`` 가 없는 경우(``pipeline.canvas`` 가 null)에는 추론조차 크기가
    고정이 아니어서 기준으로 삼을 값이 없다. 그때는 데이터의 **최다 크기**로
    맞춘다 — 배치가 되게 하는 것이 우선이고, 그 선택이 임의라는 사실은 호출자가
    경고로 알린다.

    Returns:
        ``(폭, 높이)``. 판단 근거가 아무것도 없으면(샘플 0개) ``None``.
    """
    if expected is not None:
        return expected
    if not sizes:
        return None
    return sizes.most_common(1)[0][0]
