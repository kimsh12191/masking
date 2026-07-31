"""탐지 전 텍스트 정규화 — 마스킹 회피 패턴 무력화.

개인정보를 일부러 필터에서 숨기려는 표기가 실제로 쓰인다.

    구1공8공4              한글 수사를 숫자 자리에 섞어 넣기
    공일공-일이삼사-오육칠팔  전부 한글 수사로 적기
    0lO-1234-5678          숫자를 닮은 영문자로 바꾸기 (호모글리프/leetspeak)
    ０１０－１２３４        전각 문자
    ①②③                   묶음 문자
    010​-1234​-5678         zero-width space 삽입
    hong(at)hana(dot)com   이메일 구분자 치환
    901231‑1234567         유니코드 하이픈 변종

각 정규식에 변형을 덧붙이는 방식은 조합 폭발로 유지가 안 된다. 대신 **매칭 전에
정규형으로 접는다.** 이러면 ``rules/detectors.py`` 의 모든 패턴이 한 번에 내성을
갖고, 체크섬 검증도 정규화된 숫자로 정상 동작한다.

``canonicalize()`` 는 정규 문자열과 함께 **원문 오프셋 매핑**을 돌려준다.
탐지 결과의 ``char_span`` 은 반드시 원문 기준이어야 한다 — 다운스트림이 원본
이미지/텍스트에 마스킹을 적용하기 때문이다.

핵심 설계: **치환은 문맥 게이트를 통과해야 적용된다.** "구"/"이"/"사"/"영" 은
한국어에서 매우 흔한 음절이다. 무조건 숫자로 바꾸면 "강남구" 가 "강남9" 가 되고
주소 탐지가 망가진다. 숫자가 섞인 덩어리에서만 접는다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "HANGUL_DIGITS",
    "LEET_DIGITS",
    "Canonical",
    "canonicalize",
    "canonical_text",
]

# --------------------------------------------------------------------------
# 치환 표
# --------------------------------------------------------------------------

#: 한글 수사 -> 숫자. 사이시옷 없는 한자 수사만 넣는다.
#:
#: 고유어 수사(하나, 둘, 셋 …)는 **일부러 제외**했다. 전화번호를 "하나둘셋넷" 으로
#: 적는 회피는 실제로 거의 없는데, 넣으면 "하나은행" 이 "1은행" 이 된다.
HANGUL_DIGITS: dict[str, str] = {
    "공": "0", "영": "0", "빵": "0",
    "일": "1",
    "이": "2",
    "삼": "3",
    "사": "4",
    "오": "5",
    "육": "6", "륙": "6",
    "칠": "7",
    "팔": "8",
    "구": "9",
}

#: 숫자를 닮은 문자 -> 숫자 (호모글리프 / leetspeak).
LEET_DIGITS: dict[str, str] = {
    "O": "0", "o": "0", "D": "0", "Q": "0", "○": "0", "〇": "0", "ᐤ": "0",
    "l": "1", "I": "1", "i": "1", "|": "1", "!": "1", "ㅣ": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "b": "6", "G": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
}

#: 눈에 보이지 않는 문자. 글자 사이에 끼워 넣어 정규식을 깨뜨리는 데 쓰인다.
#:
#: 리터럴로 적으면 편집기·diff·터미널을 거치며 조용히 사라지거나 뭉개진다.
#: 반드시 코드포인트로 적을 것.
_INVISIBLE = frozenset(
    [
        "\u00ad",  # SOFT HYPHEN
        "\u061c",  # ARABIC LETTER MARK
        "\u180e",  # MONGOLIAN VOWEL SEPARATOR
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u2060",  # WORD JOINER
        "\u2061",  # FUNCTION APPLICATION
        "\u2062",  # INVISIBLE TIMES
        "\u2063",  # INVISIBLE SEPARATOR
        "\u2064",  # INVISIBLE PLUS
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE (BOM)
        "\u115f",  # HANGUL CHOSEONG FILLER
        "\u1160",  # HANGUL JUNGSEONG FILLER
        "\u3164",  # HANGUL FILLER
        "\uffa0",  # HALFWIDTH HANGUL FILLER
    ]
)

#: 하이픈 변종 -> ASCII 하이픈. OCR 과 워드프로세서가 제멋대로 바꿔 놓는다.
_HYPHENS = dict.fromkeys(
    [
        "\u2010",  # HYPHEN
        "\u2011",  # NON-BREAKING HYPHEN
        "\u2012",  # FIGURE DASH
        "\u2013",  # EN DASH
        "\u2014",  # EM DASH
        "\u2015",  # HORIZONTAL BAR
        "\u2212",  # MINUS SIGN
        "\u2043",  # HYPHEN BULLET
        "\ufe63",  # SMALL HYPHEN-MINUS
        "\uff0d",  # FULLWIDTH HYPHEN-MINUS
        "\u30fc",  # KATAKANA-HIRAGANA PROLONGED SOUND MARK
    ],
    "-",
)

#: 공백 변종 -> ASCII 공백.
_SPACES = dict.fromkeys(
    [
        "\u00a0",  # NO-BREAK SPACE
        "\u1680",  # OGHAM SPACE MARK
        "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005",
        "\u2006", "\u2007", "\u2008", "\u2009", "\u200a",
        "\u202f",  # NARROW NO-BREAK SPACE
        "\u205f",  # MEDIUM MATHEMATICAL SPACE
        "\u3000",  # IDEOGRAPHIC SPACE
    ],
    " ",
)

#: 이메일 ``@`` 를 대체하는 표기.
_AT_ALT = r"\(\s*at\s*\)|\[\s*at\s*\]|\{\s*at\s*\}|골뱅이|앳|\s+at\s+"

#: 이메일 ``.`` 를 대체하는 표기.
_DOT_ALT = r"\(\s*dot\s*\)|\[\s*dot\s*\]|\{\s*dot\s*\}|\s+dot\s+|\(\s*점\s*\)|\[\s*점\s*\]"

_LOCAL = r"[A-Za-z0-9._%+\-]{1,64}"
_HOST = r"[A-Za-z0-9.\-]{1,255}"

#: 난독화된 이메일. ``(패턴, 치환식)`` 쌍이다.
#:
#: **전체 모양이 이메일일 때만** 치환한다. ``at``/``점`` 을 단독으로 치환하면
#: "지점", 영어 산문의 "at" 까지 망가진다. 앞뒤가 이메일 구조를 이루는 경우로
#: 한정해야 안전하다.
_EMAIL_OBFUSCATED: list[tuple[re.Pattern[str], str]] = [
    # local (at) host (dot) tld  — @ 와 . 둘 다 난독화
    (
        re.compile(
            rf"({_LOCAL})\s*(?:{_AT_ALT})\s*([A-Za-z0-9\-]{{1,255}})"
            rf"\s*(?:{_DOT_ALT})\s*([A-Za-z]{{2,24}})"
        ),
        r"\1@\2.\3",
    ),
    # local (at) host.tld  — @ 만 난독화
    (re.compile(rf"({_LOCAL})\s*(?:{_AT_ALT})\s*({_HOST}\.[A-Za-z]{{2,24}})"), r"\1@\2"),
]

#: 치환 게이트: 덩어리에서 구분자를 뺀 실질 길이 하한.
_MIN_TOKEN_LEN = 4

#: 치환 게이트: 실질 문자 중 "숫자이거나 숫자로 접힐 수 있는" 비율 하한.
_MIN_DIGITISH_RATIO = 0.9

#: 한글 수사만으로 이루어진 덩어리를 접으려면 이 길이 이상이어야 한다.
#:
#: "일이삼사" 는 우연히 나올 수 있지만 "공일공일이삼사오육칠팔" 은 전화번호다.
_MIN_PURE_HANGUL_RUN = 7

#: 게이트 계산에서 무시할 구분자 (표기마다 달라 실질 문자가 아니다).
_TOKEN_SEPARATORS = frozenset("-.·:()[]{}/\\_,")


# --------------------------------------------------------------------------
# 정규형 + 오프셋 매핑
# --------------------------------------------------------------------------


@dataclass
class Canonical:
    """정규화 결과.

    Attributes:
        text: 정규화된 문자열.
        starts: ``text[i]`` 가 유래한 원문 시작 오프셋.
        ends: ``text[i]`` 가 유래한 원문 끝 오프셋 (배타적).
        changed: 원문과 달라졌는가. 감사 로그용.
    """

    text: str
    starts: list[int]
    ends: list[int]
    changed: bool

    def span(self, start: int, end: int) -> tuple[int, int]:
        """정규형 구간 ``[start, end)`` 를 원문 구간으로 되돌린다."""
        if not self.starts or start >= end:
            return (0, 0)
        lo = max(0, min(start, len(self.starts) - 1))
        hi = max(0, min(end - 1, len(self.ends) - 1))
        return (self.starts[lo], self.ends[hi])


#: 내부 표현: (문자, 원문시작, 원문끝)
_Item = tuple[str, int, int]


def _initial_items(text: str) -> tuple[list[_Item], bool]:
    """NFKC 정규화 + 비가시문자 제거를 문자 단위로 수행한다.

    문자 단위로 NFKC 를 적용해야 오프셋 매핑을 유지할 수 있다. 한 문자가 여러
    문자로 풀리면(``⑴`` -> ``(1)``) 모두 같은 원문 구간을 가리킨다.
    """
    items: list[_Item] = []
    changed = False

    for i, ch in enumerate(text):
        if ch in _INVISIBLE:
            changed = True
            continue

        mapped = _HYPHENS.get(ch) or _SPACES.get(ch)
        if mapped is None:
            mapped = unicodedata.normalize("NFKC", ch)
        if mapped != ch:
            changed = True
        for out in mapped:
            if out in _INVISIBLE:
                changed = True
                continue
            items.append((out, i, i + 1))

    return items, changed


def _replace_regex(
    items: list[_Item], pattern: re.Pattern[str], repl: str
) -> tuple[list[_Item], bool]:
    """정규형 위에서 정규식 치환하고 오프셋 매핑을 유지한다.

    치환 결과 문자들은 매치 구간 전체의 원문 범위를 가리킨다 — 여러 문자가
    하나로 접히면 어느 한 글자에 귀속시킬 수 없기 때문이다.
    """
    text = "".join(it[0] for it in items)
    out: list[_Item] = []
    cursor = 0
    changed = False

    for m in pattern.finditer(text):
        if m.start() < cursor:
            continue
        out.extend(items[cursor : m.start()])
        span_start = items[m.start()][1]
        span_end = items[m.end() - 1][2]
        rendered = m.expand(repl)
        for ch in rendered:
            out.append((ch, span_start, span_end))
        cursor = m.end()
        changed = True

    out.extend(items[cursor:])
    return out, changed


def _digitish(ch: str, table: dict[str, str]) -> bool:
    return ch.isdigit() or ch in table


def _should_fold(chunk: str, table: dict[str, str], pure_needs_run: bool) -> bool:
    """이 덩어리에 숫자 치환을 적용할지 판단한다 — 문맥 게이트.

    Args:
        chunk: 공백으로 끊은 덩어리.
        table: 치환 표.
        pure_needs_run: 숫자가 하나도 없는(전부 치환 대상인) 덩어리를 접으려면
            길이 하한을 요구할지. 한글 수사에는 ``True`` 여야 한다 —
            "일이삼사" 를 접으면 오탐이지만 "공일공일이삼사오육칠팔" 은 번호다.

    Returns:
        접어도 되는가.
    """
    core = [ch for ch in chunk if ch not in _TOKEN_SEPARATORS and not ch.isspace()]
    if len(core) < _MIN_TOKEN_LEN:
        return False

    n_digit = sum(ch.isdigit() for ch in core)
    n_sub = sum(ch in table for ch in core)
    if n_digit + n_sub < _MIN_TOKEN_LEN:
        return False
    if (n_digit + n_sub) / len(core) < _MIN_DIGITISH_RATIO:
        return False

    if n_digit == 0:
        # 숫자가 전혀 없다 = 전부 치환 대상이다. 우연일 수 있으므로 길게 요구한다.
        return not pure_needs_run or len(core) >= _MIN_PURE_HANGUL_RUN
    return True


def _fold_digits(
    items: list[_Item], table: dict[str, str], pure_needs_run: bool
) -> tuple[list[_Item], bool]:
    """게이트를 통과한 덩어리에서만 문자를 숫자로 접는다."""
    text = "".join(it[0] for it in items)
    if not any(ch in table for ch in text):
        return items, False

    out = list(items)
    changed = False

    # 공백으로 덩어리를 끊는다. 하이픈·점은 표기 변형이므로 덩어리를 끊지 않는다
    # ("0lO-1234-5678" 을 세 조각으로 쪼개면 게이트를 통과할 수 없다).
    for m in re.finditer(r"\S+", text):
        chunk = m.group(0)
        if not any(ch in table for ch in chunk):
            continue
        if not _should_fold(chunk, table, pure_needs_run):
            continue
        for offset, ch in enumerate(chunk):
            if ch in table:
                pos = m.start() + offset
                _, s, e = out[pos]
                out[pos] = (table[ch], s, e)
                changed = True

    return out, changed


def canonicalize(text: str) -> Canonical:
    """회피 표기를 접은 정규형과 원문 오프셋 매핑을 만든다.

    적용 순서 (앞의 결과 위에 뒤가 적용된다):

    1. 문자 단위 NFKC — 전각/묶음문자/상단첨자/수학기호 숫자
    2. 비가시문자 제거 — zero-width space, soft hyphen 등
    3. 하이픈·공백 변종 통일
    4. 난독화된 이메일 복원 — ``(at)``, ``골뱅이``, ``(dot)``
    5. 한글 수사 -> 숫자 (문맥 게이트 통과 시)
    6. 숫자 닮은 문자 -> 숫자 (문맥 게이트 통과 시)

    Args:
        text: OCR 원문.

    Returns:
        ``Canonical``. ``changed`` 가 ``False`` 면 원문과 동일하다.
    """
    if not text:
        return Canonical("", [], [], False)

    items, changed = _initial_items(text)

    for pattern, repl in _EMAIL_OBFUSCATED:
        items, hit = _replace_regex(items, pattern, repl)
        changed = changed or hit

    items, hit = _fold_digits(items, HANGUL_DIGITS, pure_needs_run=True)
    changed = changed or hit

    items, hit = _fold_digits(items, LEET_DIGITS, pure_needs_run=False)
    changed = changed or hit

    return Canonical(
        text="".join(it[0] for it in items),
        starts=[it[1] for it in items],
        ends=[it[2] for it in items],
        changed=changed,
    )


def canonical_text(text: str) -> str:
    """오프셋 매핑이 필요 없을 때 쓰는 간편 버전."""
    return canonicalize(text).text


def identity(text: str) -> Canonical:
    """원문을 그대로 담은 ``Canonical``.

    호출부가 "원문 스캔"과 "정규형 스캔"을 같은 코드로 돌릴 수 있게 한다.
    정규화는 **대체 경로가 아니라 추가 경로**여야 한다 — 접기가 정상적인 값을
    망가뜨릴 수 있기 때문이다 (여권번호 ``S12345678`` 의 ``S`` 가 ``5`` 로
    접히면 패턴이 깨진다).
    """
    return Canonical(
        text=text,
        starts=list(range(len(text))),
        ends=list(range(1, len(text) + 1)),
        changed=False,
    )
