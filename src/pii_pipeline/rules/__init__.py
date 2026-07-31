"""체크섬 검증기.

이 패키지에는 더 이상 **탐지기가 없다.** 정규식 탐지 레이어를 제거하고
체크섬만 남겼다 — 체크섬은 VLM 이 뱉은 값을 사후 **검증**하는 데 쓰인다
(``pii_pipeline.verify``). 자리수·구성 형식 지식은 프롬프트의 형식표로 옮겼다
(``pii_pipeline.llm.prompts``).

이유: 정규식이 탐지기일 때 자리수만 맞으면 확정 라벨을 붙였고, 체크섬이 없는
라벨(운전면허번호 등)까지 "통과" 로 취급해 주민등록번호를 오분류했다.
검증기는 최악의 경우 "모르겠다" 를 낼 뿐, 틀린 답을 만들어내지 않는다.
"""

from .checksums import (
    VALIDATORS,
    digits_only,
    validate_biz_no,
    validate_corp_no,
    validate_foreign_id,
    validate_luhn,
    validate_rrn,
)

__all__ = [
    "VALIDATORS",
    "digits_only",
    "validate_rrn",
    "validate_foreign_id",
    "validate_biz_no",
    "validate_corp_no",
    "validate_luhn",
]
