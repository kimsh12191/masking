"""① VLM 탐지 단계 — 이미지를 보고 개인정보를 전사한다.

이 단계의 출력은 ``VlmFinding`` 목록이다. **좌표는 대략이고, 값이 정확하다.**
정밀 좌표는 ``locate.py`` 가 크롭 OCR 로 확정한다.

타일링을 하는 이유:

한 번의 호출로 페이지 전체를 전사하게 하면 뒤쪽 항목이 잘린다. 모델이 분류를
못해서가 아니라 **긴 구조화 출력의 후반부가 무너지기 때문**이다. 가족관계증명서
한 장에 구성원 9명 × 4필드 = 36건인데, 9B 모델이 36건을 한 번에 온전히 뱉기를
기대하는 것은 무리다.

그래서 페이지를 긴 축 방향으로 몇 조각으로 나눠 조각당 한 번씩 호출한다.
호출당 10건 내외면 안정적이다. 부수 효과로 조각을 리사이즈했을 때의 실효
해상도가 올라간다 — 같은 ``image_max_side`` 예산 안에서 글자가 더 커진다.

경계에 걸친 값을 잃지 않도록 조각을 **겹쳐서** 자르고, 겹침 때문에 생기는
중복은 ``_dedup`` 이 정리한다. 겹침 없이 자르면 경계에 걸친 한 줄이 양쪽에서
반씩 잘려 양쪽 모두 못 읽는다.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .llm.client import LlmClient
from .llm.prompts import SYSTEM_VLM, build_user
from .normalize import canonical_text
from .schema import VLM_SCHEMA, VlmFinding

log = logging.getLogger(__name__)

#: 정규화 비교 시 제거할 구분자. OCR/VLM 마다 하이픈·공백 처리가 다르다.
_STRIP = " \t\n-–—_./\\,·:;()[]{}'\"|"

#: bbox 가 퇴화(면적 0)했을 때 부여할 최소 정규화 크기.
_MIN_SIDE = 0.004

#: 동시 호출 상한. vLLM 서버 한 대를 상대로 이보다 늘려도 처리량이 늘지 않고
#: 큐만 길어진다 (배치는 서버가 알아서 묶는다).
_MAX_WORKERS = 8


@dataclass
class DetectConfig:
    """VLM 탐지 설정.

    Attributes:
        tiles: 페이지를 몇 조각으로 나눠 호출할지. ``1`` 이면 전체를 1회 호출한다.
            밀집한 표 문서(등본·가족관계증명서)는 3~4 를 권한다. 조각 수만큼
            호출이 늘지만 시스템 프롬프트는 고정이라 prefix caching 이 걸린다.
        overlap: 조각 간 겹침 비율 (조각 두께 기준). 경계에 걸친 한 줄을
            양쪽에서 반씩 잘라 둘 다 못 읽는 것을 막는다.
        dedup: 겹침 구간의 중복을 제거할지. 끄면 같은 값이 두 번 보고된다.
        hint: user 메시지에 덧붙일 문서 종류 힌트. 비워 두는 것이 기본이다.
        samples: 타일당 호출 횟수. **학습 없이 미탐을 줄이는 유일한 손잡이다.**
            2 이상이면 같은 타일을 여러 번 뽑아 ``_dedup`` 이 합집합을 만든다.
            한 번은 놓치고 다른 번엔 잡히는 항목이 회수된다 — 재현율이 오르고
            정밀도가 떨어진다. 마스킹에서는 미탐이 과탐보다 훨씬 비싸므로
            대체로 맞는 거래지만, 호출 수가 ``tiles × samples`` 로 늘어난다.
        sample_temperature: ``samples >= 2`` 일 때 쓸 온도. **0 이면 안 된다** —
            같은 입력에 같은 답이 와서 샘플을 늘린 의미가 없어진다.
            ``samples == 1`` 이면 무시되고 설정값(0)이 쓰인다 (결정론 유지).
        workers: 동시 호출 수. 0 이면 호출 수만큼 (``_MAX_WORKERS`` 상한).
            타일 호출은 서로 독립이므로 직렬로 돌릴 이유가 없다.
    """

    tiles: int = 3
    overlap: float = 0.08
    dedup: bool = True
    hint: str = ""
    samples: int = 1
    sample_temperature: float = 0.3
    workers: int = 0


# --------------------------------------------------------------------------
# 타일 기하
# --------------------------------------------------------------------------


def tile_rects(
    width: int, height: int, tiles: int, overlap: float
) -> list[tuple[float, float, float, float]]:
    """페이지를 **긴 축 방향으로** 나눈 정규화 사각형 목록을 만든다.

    긴 축을 자르는 것은 조각을 정사각형에 가깝게 만들기 위한 것이다. VLM 의
    비전 인코더는 극단적으로 긴 이미지에서 가로세로 한쪽을 크게 줄인다.

    Args:
        width: 페이지 폭 (px).
        height: 페이지 높이 (px).
        tiles: 조각 수. 1 이하면 전체 1장.
        overlap: 겹침 비율 (0.0~0.5). 조각 두께의 이 비율만큼 양쪽으로 넓힌다.

    Returns:
        ``(x1, y1, x2, y2)`` 정규화 사각형 목록. 항상 최소 1개.
    """
    if tiles <= 1:
        return [(0.0, 0.0, 1.0, 1.0)]

    overlap = min(max(overlap, 0.0), 0.5)
    band = 1.0 / tiles
    pad = band * overlap
    vertical = height >= width  # 세로가 길면 y 를 자른다

    rects: list[tuple[float, float, float, float]] = []
    for i in range(tiles):
        lo = max(0.0, i * band - pad)
        hi = min(1.0, (i + 1) * band + pad)
        rects.append((0.0, lo, 1.0, hi) if vertical else (lo, 0.0, hi, 1.0))
    return rects


def crop_norm(image: Any, rect: tuple[float, float, float, float]) -> Any:
    """정규화 사각형으로 이미지를 잘라낸다 (numpy BGR 배열 가정)."""
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(rect[0] * w)))
    y1 = max(0, min(h - 1, int(rect[1] * h)))
    x2 = max(x1 + 1, min(w, int(round(rect[2] * w))))
    y2 = max(y1 + 1, min(h, int(round(rect[3] * h))))
    return image[y1:y2, x1:x2]


def _to_page(
    local: tuple[float, float, float, float], rect: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """타일 내 정규화 좌표를 **페이지** 정규화 좌표로 환산한다.

    모델은 자기가 받은 이미지 기준으로만 답한다. 타일 번호를 프롬프트에 넣어
    모델이 환산하게 하면 틀리고, 시스템 프롬프트도 가변이 되어 캐싱이 깨진다.
    환산은 코드가 한다.
    """
    rx, ry = rect[0], rect[1]
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    x1, y1, x2, y2 = local
    return (rx + x1 * rw, ry + y1 * rh, rx + x2 * rw, ry + y2 * rh)


def _sane_bbox(
    raw: Any,
) -> tuple[float, float, float, float] | None:
    """모델이 준 bbox 를 정리한다. 못 쓰면 ``None``.

    좌표 순서가 뒤집힌 것(x2<x1)은 정렬해 살린다 — 값 자체는 맞는데 순서만
    틀린 경우가 흔하고, 버리면 탐지 하나를 잃는다. 면적이 0 이면 최소 크기를
    준다 (한 점을 찍은 경우. 패딩을 붙이면 쓸 만한 크롭이 된다).
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [min(max(float(v), 0.0), 1.0) for v in raw]
    except (TypeError, ValueError):
        return None

    x1, x2 = sorted((vals[0], vals[2]))
    y1, y2 = sorted((vals[1], vals[3]))
    if x2 - x1 < _MIN_SIDE:
        x1, x2 = max(0.0, x1 - _MIN_SIDE), min(1.0, x1 + _MIN_SIDE)
    if y2 - y1 < _MIN_SIDE:
        y1, y2 = max(0.0, y1 - _MIN_SIDE), min(1.0, y1 + _MIN_SIDE)
    return (x1, y1, x2, y2)


# --------------------------------------------------------------------------
# 중복 제거
# --------------------------------------------------------------------------


def match_key(text: str) -> str:
    """비교용 키. 회피 표기를 접고 구분자·공백을 지운다.

    ``"공1공-1234-5678"`` 과 ``"010 1234 5678"`` 이 같은 키가 되어야
    겹침 구간에서 두 번 잡힌 같은 값을 하나로 볼 수 있다.
    """
    folded = canonical_text(text)
    return "".join(ch for ch in folded if ch not in _STRIP).upper()


def _overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _dedup(findings: list[VlmFinding]) -> list[VlmFinding]:
    """겹침 구간에서 두 번 보고된 같은 값을 하나로 합친다.

    **값이 같아도 위치가 겹치지 않으면 남긴다.** 같은 이름이 문서 여러 곳에
    나오는 것은 정상이고, 그 위치를 각각 마스킹해야 한다. 텍스트만 보고
    합치면 두 번째 위치가 마스킹되지 않는다.
    """
    kept: list[VlmFinding] = []
    for item in sorted(findings, key=lambda f: (-f.conf, f.tile)):
        key = match_key(item.text)
        duplicate = any(
            k.type == item.type
            and match_key(k.text) == key
            and _overlaps(k.bbox_norm, item.bbox_norm)
            for k in kept
        )
        if not duplicate:
            kept.append(item)
    # 읽기 순서(위 -> 아래, 왼 -> 오른)로 돌려준다
    kept.sort(key=lambda f: (round(f.bbox_norm[1], 3), f.bbox_norm[0]))
    return kept


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------


def detect(
    image: Any,
    client: LlmClient,
    config: DetectConfig | None = None,
    warnings: list[str] | None = None,
) -> tuple[list[VlmFinding], list[dict[str, Any]]]:
    """이미지에서 개인정보를 전사한다.

    Args:
        image: 전처리된 BGR numpy 배열.
        client: vLLM 클라이언트.
        config: 탐지 설정.
        warnings: 경고를 append 할 목록.

    Returns:
        ``(탐지 목록, 호출별 메타정보)``. 한 타일이 실패해도 나머지는 유지한다 —
        페이지 하나를 통째로 버리는 것이 최악이다. 실패한 타일은 메타에
        ``error`` 로 남고 ``warnings`` 에도 기록된다.
    """
    cfg = config or DetectConfig()
    warn = warnings if warnings is not None else []
    h, w = image.shape[:2]

    rects = tile_rects(w, h, cfg.tiles, cfg.overlap)
    user = build_user(cfg.hint)
    samples = max(1, cfg.samples)
    # samples==1 이면 온도를 건드리지 않는다 — 기존의 결정론적 동작을 지킨다.
    temp = None if samples == 1 else cfg.sample_temperature

    # (타일, 샘플) 조합. 순서를 고정해 두면 병렬로 돌려도 결과가 결정론적이다.
    calls = [(t, s) for t in range(len(rects)) for s in range(samples)]
    tile_imgs = [
        crop_norm(image, rect) if len(rects) > 1 else image for rect in rects
    ]

    def one(job: tuple[int, int]) -> dict[str, Any]:
        tile_no, sample_no = job
        payload, meta = client.complete_json(
            system=SYSTEM_VLM,
            user=user,
            schema=VLM_SCHEMA,
            image=tile_imgs[tile_no],
            temperature=temp,
        )
        meta["tile"] = tile_no
        meta["sample"] = sample_no
        meta["rect"] = [round(v, 4) for v in rects[tile_no]]
        meta["payload"] = payload
        return meta

    if len(calls) == 1:
        metas = [one(calls[0])]
    else:
        # 클라이언트를 미리 만들어 둔다 (지연 초기화가 스레드에서 겹치지 않게).
        _ = client.client
        n_workers = cfg.workers or min(len(calls), _MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            metas = list(pool.map(one, calls))

    findings: list[VlmFinding] = []
    for meta in metas:
        tile_no = meta["tile"]
        rect = rects[tile_no]
        where = f"타일 {tile_no}" + (
            f" 샘플 {meta['sample']}" if samples > 1 else ""
        )
        payload = meta.pop("payload", {}) or {}

        if meta.get("error"):
            warn.append(f"VLM {where} 실패: {meta['error']}")
            continue
        if meta.get("salvaged"):
            warn.append(
                f"VLM {where}: JSON 형식 이탈을 복구했습니다 "
                f"({meta['salvaged']}) — 프롬프트/guided decoding 점검 필요"
            )

        n_before = len(findings)
        for item in payload.get("findings") or []:
            if not isinstance(item, dict):
                continue
            label = item.get("type")
            if not label:
                continue
            bbox = _sane_bbox(item.get("bbox_norm"))
            if bbox is None:
                warn.append(
                    f"VLM {where}: bbox_norm 이 없거나 잘못된 항목 폐기 "
                    f"(type={label}, text={str(item.get('text'))[:20]!r})"
                )
                continue
            findings.append(
                VlmFinding(
                    text=str(item.get("text") or ""),
                    type=str(label),
                    field=str(item.get("field") or ""),
                    bbox_norm=_to_page(bbox, rect),
                    conf=float(item.get("conf") or 0.0),
                    tile=tile_no,
                )
            )
        log.debug("%s: %d건", where, len(findings) - n_before)

    if not cfg.dedup:
        return findings, metas

    deduped = _dedup(findings)
    if len(deduped) < len(findings):
        log.debug(
            "중복 %d건 병합 (타일 겹침%s)",
            len(findings) - len(deduped),
            " + 다중 샘플" if samples > 1 else "",
        )
    return deduped, metas
