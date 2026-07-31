#!/usr/bin/env python3
"""합성 금융 서식 생성기 (검증용).

실제 내부 자료 없이 파이프라인을 검증하기 위한 가짜 문서를 만든다.
**정답(ground truth) JSON 을 함께 출력**하므로 recall/precision 측정에 바로 쓸 수 있다.

의도적으로 다음 난이도를 섞는다:

* 손글씨 흉내 (기울어진 다른 폰트) — OCR rec 이 자주 실패한다
* 도장/인영 (반투명 빨간 원) — 글자를 가린다
* 저대비 작은 글씨 — LOW_CONF 를 유발한다
* 다단 컬럼 — 읽기 순서 정렬을 검증한다
* **인라인 라벨** ("담당자: 조민석") — 라벨과 값이 한 OCR 박스에 섞인 형태.
  실제 문서에 흔하지만 라벨/값이 분리된 서식만 만들면 이 케이스를 놓친다.

사용 예:

    python scripts/make_synthetic.py --out data/synth --count 5
    python scripts/run.py data/synth/*.png -o out/ --overlay
    python scripts/eval.py out/ data/synth/    # (평가 스크립트는 별도 작성)

주의: 여기서 만드는 번호는 모두 체크섬 알고리즘으로 합성한 가짜 값이다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pii_pipeline.rules.checksums import (  # noqa: E402
    _BIZ_WEIGHTS,
    _RRN_WEIGHTS,
)

PAGE_W, PAGE_H = 1748, 2480  # A4 210dpi

#: 한글 렌더링이 가능한 폰트 후보. 앞에서부터 찾는다.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]

SURNAMES = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
GIVEN = ["민준", "서연", "지호", "하은", "예준", "수아", "도윤", "지우", "건우", "채원"]
CITIES = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "경기도"]
DISTRICTS = ["강남구", "서초구", "송파구", "영등포구", "마포구", "성남시 분당구"]
ROADS = ["테헤란로", "강남대로", "여의대로", "월드컵로", "판교역로"]
BUILDINGS = ["○○빌딩", "△△타워", "□□센터", "◇◇프라자"]
DEPTS = ["리스크관리부", "여신심사팀", "IT개발본부", "자금운용팀", "준법감시실"]
BANKS = ["하나은행", "국민은행", "신한은행", "우리은행", "농협은행"]
TITLES = ["과장", "차장", "부장", "팀장", "대리", "사원"]
PRODUCTS = ["신용대출", "주택담보대출", "전세자금대출", "사업자대출"]
#: 실제 유선 지역번호
AREA_CODES = ["02", "031", "032", "033", "041", "042", "043", "051",
              "052", "053", "054", "055", "061", "062", "063", "064"]


# --------------------------------------------------------------------------
# 유효한 체크섬 번호 생성
# --------------------------------------------------------------------------


def gen_rrn(rng: random.Random, foreign: bool = False) -> str:
    yy = rng.randint(60, 99)
    mm = rng.randint(1, 12)
    dd = rng.randint(1, 28)
    gender = rng.choice([5, 6, 7, 8] if foreign else [1, 2, 3, 4])
    # 앞 12자리 = 생년월일(6) + 성별코드(1) + 일련번호(5). 마지막 1자리가 체크디짓.
    p12 = f"{yy:02d}{mm:02d}{dd:02d}{gender}{rng.randint(0, 99999):05d}"
    total = sum(int(c) * w for c, w in zip(p12, _RRN_WEIGHTS, strict=True))
    return f"{p12[:6]}-{p12[6:]}{(11 - total % 11) % 10}"


def gen_biz_no(rng: random.Random) -> str:
    # 형식은 3-2-5 (총 10자리)이고 마지막 1자리가 체크디짓이므로 앞 9자리를 만든다.
    d9 = f"{rng.randint(100, 999)}{rng.randint(10, 99)}{rng.randint(0, 9999):04d}"
    total = sum(int(c) * w for c, w in zip(d9, _BIZ_WEIGHTS, strict=True))
    total += (int(d9[8]) * 5) // 10
    d10 = d9 + str((10 - total % 10) % 10)
    return f"{d10[:3]}-{d10[3:5]}-{d10[5:]}"


def gen_card(rng: random.Random) -> str:
    d15 = f"{rng.choice([4, 5])}" + f"{rng.randint(0, 10**14 - 1):014d}"
    total = 0
    for i, ch in enumerate(reversed(d15 + "0")):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    check = (10 - total % 10) % 10
    d16 = d15 + str(check)
    return "-".join(d16[i : i + 4] for i in range(0, 16, 4))


def gen_phone(rng: random.Random) -> str:
    return f"010-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"


def gen_account(rng: random.Random) -> str:
    return f"{rng.randint(100, 999)}-{rng.randint(100000, 999999)}-{rng.randint(10, 99)}"


def gen_email(rng: random.Random, name_ascii: str) -> str:
    domain = rng.choice(["example.com", "test.co.kr", "sample.or.kr"])
    return f"{name_ascii}{rng.randint(1, 99)}@{domain}"


def gen_ip(rng: random.Random) -> str:
    """사설 IP 대역에서 생성한다 (실제 공인 IP 를 만들지 않기 위해)."""
    kind = rng.choice(["10", "172", "192"])
    if kind == "10":
        return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    if kind == "172":
        return f"172.{rng.randint(16, 31)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
    return f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def gen_passport(rng: random.Random) -> str:
    return f"{rng.choice('MSR')}{rng.randint(0, 99999999):08d}"


# --------------------------------------------------------------------------
# 렌더링
# --------------------------------------------------------------------------


def find_font() -> str:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return path
    raise SystemExit(
        "한글 렌더링이 가능한 폰트를 찾지 못했습니다.\n"
        "NanumGothic 또는 Noto Sans CJK 를 설치하거나 --font 로 경로를 지정하십시오."
    )


def make_page(
    rng: random.Random, font_path: str, page_no: int
) -> tuple[object, list[dict[str, object]]]:
    """서식 1장과 정답 목록을 생성한다."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (PAGE_W, PAGE_H), (252, 252, 250))
    draw = ImageDraw.Draw(img, "RGBA")

    f_title = ImageFont.truetype(font_path, 46)
    f_label = ImageFont.truetype(font_path, 30)
    f_value = ImageFont.truetype(font_path, 30)
    f_small = ImageFont.truetype(font_path, 22)

    truth: list[dict[str, object]] = []

    def put(x: int, y: int, text: str, font, fill=(20, 20, 20), label: str | None = None):
        draw.text((x, y), text, font=font, fill=fill)
        bbox = draw.textbbox((x, y), text, font=font)
        if label:
            truth.append({"type": label, "bbox": list(bbox), "text": text})
        return bbox

    # 제목 / 테두리
    put(PAGE_W // 2 - 220, 90, "여 신 거 래 약 정 서", f_title)
    draw.rectangle((80, 60, PAGE_W - 80, PAGE_H - 80), outline=(150, 150, 150), width=2)
    draw.line((80, 180, PAGE_W - 80, 180), fill=(150, 150, 150), width=2)

    surname, given = rng.choice(SURNAMES), rng.choice(GIVEN)
    name = surname + given
    name_ascii = f"user{rng.randint(100, 999)}"

    left_x, value_x, y = 130, 520, 240
    step = 78

    rows: list[tuple[str, str, str | None]] = [
        ("성명", name, "NAME"),
        ("주민등록번호", gen_rrn(rng), "RRN"),
        ("연락처", gen_phone(rng), "PHONE"),
        ("전자우편", gen_email(rng, name_ascii), "EMAIL"),
        ("직장명", f"{rng.choice(BANKS)} {rng.choice(DEPTS)}", "ORG"),
        ("직위", rng.choice(TITLES), "TITLE"),
        ("사업자등록번호", gen_biz_no(rng), "BIZ_NO"),
        ("여권번호", gen_passport(rng), "PASSPORT"),
        ("결제카드번호", gen_card(rng), "CARD_NO"),
        ("입금계좌번호", f"{rng.choice(BANKS)} {gen_account(rng)}", "ACCOUNT_NO"),
        ("전자서명 접속IP", gen_ip(rng), "IP"),
    ]

    for label_text, value, pii_type in rows:
        put(left_x, y, label_text, f_label, fill=(70, 70, 70))
        put(value_x, y, value, f_value, label=pii_type)
        y += step

    # 주소: 3개 박스로 쪼개어 배치 (병합 로직 검증용)
    put(left_x, y, "주소", f_label, fill=(70, 70, 70))
    a1 = f"{rng.choice(CITIES)} {rng.choice(DISTRICTS)}"
    a2 = f"{rng.choice(ROADS)} {rng.randint(1, 400)}"
    a3 = f"{rng.choice(BUILDINGS)} {rng.randint(2, 20)}층"
    b1 = draw.textbbox((value_x, y), a1, font=f_value)
    draw.text((value_x, y), a1, font=f_value, fill=(20, 20, 20))
    b2 = draw.textbbox((b1[2] + 24, y), a2, font=f_value)
    draw.text((b1[2] + 24, y), a2, font=f_value, fill=(20, 20, 20))
    b3 = draw.textbbox((value_x, y + 44), a3, font=f_value)
    draw.text((value_x, y + 44), a3, font=f_value, fill=(20, 20, 20))
    truth.append(
        {
            "type": "ADDRESS",
            "bbox": [
                min(b1[0], b3[0]), min(b1[1], b3[1]),
                max(b2[2], b3[2]), max(b2[3], b3[3]),
            ],
            "text": f"{a1} {a2} {a3}",
            "member_boxes": [list(b1), list(b2), list(b3)],
        }
    )
    y += 130

    # 개인정보가 아닌 항목 (과검 측정용)
    put(left_x, y, "대출상품", f_label, fill=(70, 70, 70))
    put(value_x, y, rng.choice(PRODUCTS), f_value)
    y += step
    put(left_x, y, "대출금액", f_label, fill=(70, 70, 70))
    put(value_x, y, f"{rng.randint(10, 300) * 1000000:,}원", f_value)
    y += step
    put(left_x, y, "약정일자", f_label, fill=(70, 70, 70))
    date_text = f"20{rng.randint(20, 26)}. {rng.randint(1, 12):02d}. {rng.randint(1, 28):02d}."
    put(value_x, y, date_text, f_value)
    y += step

    # 저대비 작은 글씨 (LOW_CONF 유발)
    put(left_x, y, "비상연락처", f_small, fill=(150, 150, 150))
    put(value_x, y, gen_phone(rng), f_small, fill=(155, 155, 155), label="PHONE")
    y += 70

    # 손글씨 흉내: 기울인 별도 레이어 (OCR rec 실패 유도)
    hand_name = rng.choice(SURNAMES) + rng.choice(GIVEN)
    put(left_x, y, "보증인 성명", f_label, fill=(70, 70, 70))
    layer = Image.new("RGBA", (420, 120), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(
        (10, 20), hand_name, font=ImageFont.truetype(font_path, 40), fill=(25, 35, 120, 255)
    )
    layer = layer.rotate(rng.uniform(-7, -3), resample=Image.BICUBIC, expand=False)
    img.paste(layer, (value_x, y - 15), layer)
    truth.append(
        {
            "type": "NAME",
            "bbox": [value_x + 10, y + 5, value_x + 190, y + 65],
            "text": hand_name,
            "difficulty": "handwriting",
        }
    )
    y += 110

    # 도장/인영: 글자를 가리는 반투명 원
    stamp_name = rng.choice(SURNAMES) + rng.choice(GIVEN)
    put(left_x, y, "확인 (인)", f_label, fill=(70, 70, 70))
    sx, sy = value_x + 40, y - 5
    put(sx, sy + 12, stamp_name, f_value, label="NAME")
    draw.ellipse(
        (sx - 12, sy - 8, sx + 130, sy + 116),
        outline=(200, 40, 40, 210),
        fill=(210, 60, 60, 70),
        width=5,
    )
    truth[-1]["difficulty"] = "stamp_overlap"
    y += 150

    # 자필 서명
    put(left_x, y, "신청인 서명", f_label, fill=(70, 70, 70))
    sig_x, sig_y = value_x, y + 10
    pts = [
        (sig_x + i * 22, sig_y + 30 + int(26 * ((-1) ** i) * rng.uniform(0.4, 1.0)))
        for i in range(11)
    ]
    draw.line(pts, fill=(20, 20, 90), width=4, joint="curve")
    truth.append(
        {
            "type": "SIGNATURE",
            "bbox": [sig_x - 5, sig_y, sig_x + 235, sig_y + 65],
            "text": None,
            "difficulty": "signature",
        }
    )

    # 증명문 블록 — 라벨과 값이 **한 줄(= 한 OCR 박스)** 에 섞인 형태.
    # 실제 행정/금융 문서에 흔하다. 라벨/값이 분리된 서식만 만들면
    # "담당자: 조민석" 같은 박스를 프롬프트가 건너뛰는 결함을 못 잡는다.
    y += 60
    draw.line((left_x, y, PAGE_W - 130, y), fill=(190, 190, 190), width=1)
    y += 30
    put(left_x, y, "위 기재사항은 원본 내용과 틀림없음을 증명합니다.", f_small,
        fill=(90, 90, 90))
    y += 50

    staff = rng.choice(SURNAMES) + rng.choice(GIVEN)
    inline_bbox = put(left_x, y, f"담당자: {staff}", f_value)
    truth.append({
        "type": "NAME", "bbox": list(inline_bbox), "text": staff,
        "difficulty": "inline_label",
        "note": "라벨과 값이 한 박스. 박스 전체가 정답 영역이다.",
    })
    tel = f"{rng.choice(AREA_CODES)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
    tel_bbox = put(left_x + 430, y, f"전화: {tel}", f_value)
    truth.append({
        "type": "PHONE", "bbox": list(tel_bbox), "text": tel,
        "difficulty": "inline_label",
        "note": "라벨과 값이 한 박스. 규칙 레이어가 값을 잡지만 bbox 는 박스 전체다.",
    })
    y += 56

    applicant = rng.choice(SURNAMES) + rng.choice(GIVEN)
    inline_bbox = put(left_x, y, f"신청인: {applicant}", f_value)
    truth.append({
        "type": "NAME", "bbox": list(inline_bbox), "text": applicant,
        "difficulty": "inline_label",
        "note": "라벨과 값이 한 박스. 박스 전체가 정답 영역이다.",
    })
    # 라벨 없는 생년월일 — 형태와 주변 문맥으로만 판단해야 한다
    birth_bbox = put(left_x + 430, y,
                     f"(19{rng.randint(60, 99)}-{rng.randint(1, 12):02d}-"
                     f"{rng.randint(1, 28):02d})", f_value)
    truth.append({
        "type": "BIRTH", "bbox": list(birth_bbox), "text": None,
        "difficulty": "unlabeled_value",
        "note": "라벨이 없다. 옆의 사람 이름을 보고 생년월일로 판단해야 한다.",
    })
    y += 56
    put(left_x, y, "용도 및 목적:", f_value, fill=(70, 70, 70))  # 라벨만 — 개인정보 아님

    put(130, PAGE_H - 150, f"문서번호 SYN-{page_no:04d} / 본 문서는 검증용 합성 데이터입니다.",
        f_small, fill=(120, 120, 120))

    return img, truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/synth", help="출력 디렉터리")
    parser.add_argument("--count", type=int, default=3, help="생성할 페이지 수")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드 (재현성)")
    parser.add_argument("--font", default=None, help="한글 폰트 경로")
    args = parser.parse_args(argv)

    try:
        import PIL  # noqa: F401
    except ImportError:
        print("Pillow 가 필요합니다: pip install Pillow", file=sys.stderr)
        return 1

    font_path = args.font or find_font()
    print(f"폰트: {font_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in range(1, args.count + 1):
        rng = random.Random(args.seed + n)
        img, truth = make_page(rng, font_path, n)
        img_path = out_dir / f"synth_{n:03d}.png"
        img.save(img_path)
        (out_dir / f"synth_{n:03d}.truth.json").write_text(
            json.dumps(
                {
                    "image": img_path.name,
                    "page": {"width": PAGE_W, "height": PAGE_H},
                    "regions": truth,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        n_hard = sum(1 for t in truth if t.get("difficulty"))
        print(f"  {img_path}  정답 {len(truth)}건 (난이도 케이스 {n_hard}건)")

    print(f"\n완료: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
