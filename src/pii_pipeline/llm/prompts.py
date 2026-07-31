"""프롬프트 구성.

시스템 프롬프트는 **고정**이다 (vLLM prefix caching 이 걸리도록).
가변 정보는 전부 user 메시지의 박스 목록에 담는다.
"""

from __future__ import annotations

from ..schema import CONTEXT_LABELS, PII_LABELS, OcrBox, OcrStatus

# --------------------------------------------------------------------------
# pass-1: OCR 텍스트만 보고 문맥 항목 분류
# --------------------------------------------------------------------------

SYSTEM_PASS1 = f"""너는 행정·금융 문서의 개인정보 영역을 식별한다.

입력은 OCR 박스 목록이며 형식은 다음과 같다.
  [번호] (x,y) 텍스트
x, y 는 페이지 크기로 정규화된 박스 좌상단 좌표다 (0.0~1.0).
`<CONFIRMED:타입>` 이 붙은 박스는 이미 확정되었으므로 다시 출력하지 마라.
`<OCR_FAILED>` 는 글자를 읽지 못한 박스, `<LOW_CONF>` 는 오독 가능성이 있는 박스다.

출력은 개인정보에 해당하는 박스의 **번호**다.
텍스트를 다시 쓰지 마라. 좌표를 추측하지 마라. 번호만 답하라.

규칙:
1. **너는 박스 단위로만 답할 수 있다.** 항목명(라벨)이 값과 같은 박스에 섞여
   있어도 그 박스를 빠뜨리지 마라. 세 경우를 구분하라.
   a) 라벨과 값이 **다른 박스**  -> 값 박스만 포함
      예: [04] "성명"  [05] "홍길동"   ->  {{"idx":[5],"type":"NAME"}}
   b) 라벨과 값이 **한 박스**    -> **그 박스를 포함**
      예: [02] "담당자: 조민석"        ->  {{"idx":[2],"type":"NAME"}}
      예: [07] "주소 서울시 강남구"    ->  {{"idx":[7],"type":"ADDRESS"}}
   c) 라벨만 있고 값이 없는 박스 -> 제외
      예: [06] "용도 및 목적:"         ->  제외
2. **라벨이 없어도 형태와 주변 문맥으로 판단하라.** 라벨이 붙어 있어야만
   개인정보인 것이 아니다.
   예: 사람 이름 옆/아래의 "(1978-04-15)" -> {{"idx":[5],"type":"BIRTH"}}
   예: 서명란 아래에 홀로 있는 사람 이름   -> NAME
3. 하나의 정보가 여러 박스에 나뉘어 있으면 idx 배열에 함께 담아라.
   예: 주소가 [06][07][08] 에 걸쳐 있으면 {{"idx":[6,7,8],"type":"ADDRESS"}}
   서로 붙어 있는 박스만 묶어라. 멀리 떨어진 박스를 묶지 마라.
4. 판단이 애매하면 **포함시켜라**. 누락이 오탐보다 위험하다.
5. type 은 다음 중 하나만 사용한다:
   {" ".join(CONTEXT_LABELS)}
6. conf 는 확신도 0.0~1.0 이다.
7. 개인정보가 없으면 {{"regions":[]}} 를 반환하라."""


# --------------------------------------------------------------------------
# pass-2: 이미지를 직접 보고 누락 회수
# --------------------------------------------------------------------------

SYSTEM_PASS2 = f"""너는 개인정보 탐지 결과를 검수하는 감사자다.
1차 탐지가 **놓친 것**을 찾는 것이 네 임무다. 이미 찾은 것을 확인하는 게 아니다.

주어지는 것:
 - 문서 이미지 (직접 보아라)
 - OCR 박스 목록. 일부는 `<OCR_FAILED>` 또는 `<LOW_CONF>` 다.
 - 1차에서 이미 탐지된 박스 번호 목록

할 일:
 이미지를 직접 보고, 개인정보인데 1차 목록에 없는 것을 찾아라.
 특히 다음을 집중해서 보아라. OCR 은 이런 것을 구조적으로 놓친다.
   - 손글씨로 기재된 항목
   - 도장·인영·서명과 그 주변에 가려진 텍스트
   - `<OCR_FAILED>` 박스의 실제 내용 (이미지에서 직접 읽어라)
   - `<LOW_CONF>` 박스의 오독 (실제로는 이름/번호일 수 있다)
   - 표 안에 묻힌 항목, 여백에 손으로 추가된 내용
   - 사진, 신분증 이미지가 붙어 있는 영역

응답 방법 — 이 순서를 반드시 지켜라:
 1. 해당 위치에 OCR 박스 번호가 **있으면** `idx` 에 그 번호를 넣어라. 이것이 기본이다.
 2. OCR 박스가 **전혀 없는** 영역일 때만 `bbox_norm` 에 [x1,y1,x2,y2] 정규화
    좌표를 넣어라. 이 경우 좌표 정확도가 떨어지므로 최소한으로만 사용하라.
 3. idx 와 bbox_norm 중 **하나만** 채워라.

**너는 박스 단위로만 답할 수 있다.** 항목명(라벨)이 값과 같은 박스에 섞여 있어도
그 박스를 빠뜨리지 마라 (예: "담당자: 조민석" -> 그 박스를 NAME 으로 보고).
라벨이 없어도 형태와 주변 문맥으로 판단하라 (예: 이름 옆의 "(1978-04-15)" -> BIRTH).

이미 탐지된 번호는 다시 보고하지 마라.
놓친 게 없으면 {{"missed":[]}} 를 반환하라. 억지로 만들어내지 마라.

type 은 다음 중 하나만 사용한다:
 {" ".join(PII_LABELS)}
reason 은 왜 개인정보라고 판단했는지 한 문장으로 짧게 적어라."""


# --------------------------------------------------------------------------
# 박스 목록 렌더러
# --------------------------------------------------------------------------

_STATUS_TAG = {
    OcrStatus.OK: "",
    OcrStatus.LOW_CONF: "<LOW_CONF>",
    OcrStatus.FAILED: "<OCR_FAILED>",
}


def render_box_list(
    boxes: list[OcrBox],
    page_w: int,
    page_h: int,
    confirmed: dict[int, str] | None = None,
) -> str:
    """OCR 박스 목록을 프롬프트 텍스트로 변환한다.

    Args:
        boxes: 읽기순서 정렬된 박스 목록.
        page_w: 페이지 폭 (정규화용).
        page_h: 페이지 높이 (정규화용).
        confirmed: ``{박스번호: 라벨}``. 규칙 레이어가 확정한 항목.
            지우지 않고 태그로 남긴다 — 필드 라벨이 보여야 모델이 서식 구조를
            이해하고 인접 항목을 제대로 판단한다.

    Returns:
        ``[번호] (x,y) 텍스트 <태그>`` 형식의 줄바꿈 구분 문자열.
    """
    confirmed = confirmed or {}
    lines: list[str] = []

    for box in boxes:
        x, y = box.norm_xy(page_w, page_h)
        text = box.text if box.text.strip() else "???"
        tags: list[str] = []

        status_tag = _STATUS_TAG[box.status]
        if status_tag:
            tags.append(status_tag)
        if box.index in confirmed:
            tags.append(f"<CONFIRMED:{confirmed[box.index]}>")

        suffix = (" " + " ".join(tags)) if tags else ""
        lines.append(f"[{box.index:02d}] ({x:.2f},{y:.2f}) {text}{suffix}")

    return "\n".join(lines)


def build_pass1_user(
    boxes: list[OcrBox],
    page_w: int,
    page_h: int,
    confirmed: dict[int, str] | None = None,
) -> str:
    return (
        "다음은 금융 문서 1장의 OCR 결과다. 개인정보에 해당하는 박스 번호를 반환하라.\n\n"
        + render_box_list(boxes, page_w, page_h, confirmed)
    )


def build_pass2_user(
    boxes: list[OcrBox],
    page_w: int,
    page_h: int,
    confirmed: dict[int, str] | None = None,
    detected_idx: list[int] | None = None,
    blind: bool = False,
) -> str:
    """pass-2 user 메시지.

    Args:
        blind: ``True`` 면 1차 탐지 결과를 감춘다. 앵커링 편향을 없애는 대신
            중복 판단이 늘어난다. 검증셋에서 두 방식을 비교해볼 것.
    """
    parts = ["다음은 금융 문서 1장이다. 이미지를 직접 보고 1차 탐지가 놓친 개인정보를 찾아라.\n"]
    parts.append("OCR 박스 목록:")
    parts.append(render_box_list(boxes, page_w, page_h, confirmed))

    if not blind:
        idx_list = sorted(detected_idx or [])
        parts.append("")
        parts.append(
            "1차에서 이미 탐지된 박스 번호: "
            + (", ".join(str(i) for i in idx_list) if idx_list else "(없음)")
        )
        parts.append("위 번호는 다시 보고하지 마라. 놓친 것만 찾아라.")
    else:
        parts.append("")
        parts.append("이미지에서 개인정보로 판단되는 모든 영역을 독립적으로 보고하라.")

    return "\n".join(parts)
