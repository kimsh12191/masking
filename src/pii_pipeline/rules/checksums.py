"""한국 식별번호 체크섬 검증.

**탐지가 아니라 검증에 쓰인다.** VLM 이 보고한 값을 ``verify.py`` 가 여기로
넘겨 검산한다. 통과하면 그 값이 실재한다는 강한 증거이므로 ``verified=True``
가 되고 사람 검토 대상에서 빠진다 (두 엔진이 같은 값을 읽었을 때).

이 구분이 중요하다. 이전 구조에서는 같은 함수들이 정규식 탐지 레이어의
**확정 근거**로 쓰였고, 검증 함수가 없는 라벨(운전면허번호 등)까지 "통과" 로
취급되어 주민등록번호를 오분류했다. 검증기는 최악의 경우 "모르겠다" 를 낼
뿐이지만, 탐지기는 틀린 답을 확신을 담아 만들어낸다.

``VALIDATORS`` 에 **없는** 라벨은 체크섬이 존재하지 않는다는 뜻이다
(전화·이메일·IP·계좌·여권·운전면허). ``verify.py`` 는 그 경우
``checksum=None`` 을 남겨 "검증 불가" 와 "검증 실패" 를 구분한다.
"""

from __future__ import annotations

_RRN_WEIGHTS = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
_BIZ_WEIGHTS = (1, 3, 7, 1, 3, 7, 1, 3, 5)
_CORP_WEIGHTS = (1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2)


def digits_only(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _valid_mmdd(month: int, day: int) -> bool:
    if not 1 <= month <= 12:
        return False
    if not 1 <= day <= 31:
        return False
    if month in (4, 6, 9, 11) and day > 30:
        return False
    return not (month == 2 and day > 29)


def validate_rrn(value: str) -> bool:
    """주민등록번호 13자리 체크섬.

    가중치 (2,3,4,5,6,7,8,9,2,3,4,5) 를 앞 12자리에 적용하고
    ``(11 - sum % 11) % 10`` 이 마지막 자리와 일치해야 한다.
    생년월일부(MMDD)와 성별코드도 함께 확인한다.
    """
    d = digits_only(value)
    if len(d) != 13:
        return False

    if not _valid_mmdd(int(d[2:4]), int(d[4:6])):
        return False

    # 성별/세기 코드는 0~9 전부 유효하므로 별도 검증하지 않는다.
    # (1~4 내국인, 5~8 외국인, 9/0 1800년대)
    total = sum(int(ch) * w for ch, w in zip(d[:12], _RRN_WEIGHTS, strict=True))
    return (11 - total % 11) % 10 == int(d[12])


def validate_foreign_id(value: str) -> bool:
    """외국인등록번호.

    2020-10 이전 발급분은 주민등록번호와 동일한 체크섬을 따른다.
    이후 발급분은 체크섬이 없으므로 형식만 확인한다 -> ``True`` 를 넓게 반환하여
    미탐을 만들지 않는다 (recall 우선).
    """
    d = digits_only(value)
    if len(d) != 13:
        return False
    if not _valid_mmdd(int(d[2:4]), int(d[4:6])):
        return False
    # 외국인 성별코드
    if d[6] not in "5678":
        return False
    if validate_rrn(d):
        return True
    # 신규 발급분: 체크섬 없음. 형식이 맞으면 통과시킨다.
    return True


def validate_biz_no(value: str) -> bool:
    """사업자등록번호 10자리 체크섬.

    가중치 (1,3,7,1,3,7,1,3,5) 적용 후 9번째 자리 곱의 십의 자리를 더하고
    ``(10 - total % 10) % 10`` 이 마지막 자리와 일치해야 한다.
    """
    d = digits_only(value)
    if len(d) != 10:
        return False
    total = sum(int(ch) * w for ch, w in zip(d[:9], _BIZ_WEIGHTS, strict=True))
    total += (int(d[8]) * 5) // 10
    return (10 - total % 10) % 10 == int(d[9])


def validate_corp_no(value: str) -> bool:
    """법인등록번호 13자리 체크섬 (가중치 1,2 교대)."""
    d = digits_only(value)
    if len(d) != 13:
        return False
    total = sum(int(ch) * w for ch, w in zip(d[:12], _CORP_WEIGHTS, strict=True))
    return (10 - total % 10) % 10 == int(d[12])


def validate_luhn(value: str) -> bool:
    """카드번호 Luhn 검증 (13~19자리)."""
    d = digits_only(value)
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


#: 라벨 -> 검증 함수. 검증 함수가 없는 라벨은 패턴만으로 확정한다.
#: (EMAIL / IP / PHONE / PASSPORT / ACCOUNT_NO 는 표준 체크섬이 없다.)
VALIDATORS = {
    "RRN": validate_rrn,
    "FOREIGN_ID": validate_foreign_id,
    "BIZ_NO": validate_biz_no,
    "CORP_NO": validate_corp_no,
    "CARD_NO": validate_luhn,
}
