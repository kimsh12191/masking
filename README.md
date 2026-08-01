# 개인정보 영역 탐지

문서 이미지(스캔·PDF)에서 개인정보의 **위치와 종류**를 찾아낸다. 마스킹 모듈이
바로 쓸 수 있는 픽셀 좌표를 낸다.

```
문서 이미지  →  [{"type":"RRN","bbox":[412,780,690,812]}, ...]  →  마스킹
```

---

## 핵심

```
1. OCR 의 좌표 정확도를 LoRA 로 VLM 에 이식한다
2. VLM 에게 개인정보를 좌표와 함께 달라고 한다
```

**왜 이렇게 하는가.** 둘 다 반쪽씩만 할 줄 안다.

| | 뭐가 개인정보인가 | 정확히 어디인가 |
|---|---|---|
| **VLM** | 잘한다 | **못한다** — 비전 토큰 1개가 32×32px 이라 300dpi 한글 한 줄(약 35px)보다 정확할 수 없다 |
| **OCR** | 모른다 | **정확하다** — 검출 박스가 곧 픽셀 좌표다 |

VLM 의 약점은 프롬프트로 못 고친다. 패치 크기는 아키텍처다. 그래서 **OCR 결과를
정답으로 삼아 VLM 을 학습**시켜 좌표 능력을 옮긴다. 라벨은 OCR 이 만드므로
사람이 붙일 것이 없다.

> **아직 학습 전이라면** 파이프라인이 크롭 OCR 로 좌표를 보정한다
> (`locate.py`). 학습된 모델을 올리면 이 보정 단계가 대부분 필요 없어진다.

---

## 빠른 시작

### 1. 의존성 없이 로직 확인

```bash
pip install pytest 'numpy<2.0' 'Pillow<11' opencv-python-headless PyYAML pypdfium2
python -m pytest          # 791 passed
python scripts/run.py --print-config
```

정답 JSON 이 딸린 합성 서식도 GPU 없이 만들 수 있다.

```bash
python scripts/make_synthetic.py --out data/synth --count 3
```

### 2. 설치 (GPU)

```bash
pip install -r requirements.txt && pip install -e .
```

`paddlepaddle-gpu` 는 CUDA 버전에 맞는 휠을 따로 깐다 (`requirements.txt` 주석).

**폐쇄망이면** PaddleOCR 모델을 미리 반입한다. 첫 실행 시 자동 다운로드가 죽는다.

```bash
python scripts/download_models.py --out ./ocr_models     # 외부망에서
export PII_OCR_DET_DIR=$PWD/ocr_models/det               # 반입 후
export PII_OCR_REC_DIR=$PWD/ocr_models/rec
export PII_OCR_CLS_DIR=$PWD/ocr_models/cls
```

### 3. vLLM 기동

```bash
vllm serve Qwen/Qwen3.5-9B \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt image=1 \
  --enable-prefix-caching \
  --guided-decoding-backend xgrammar
```

`--max-model-len 8192` 를 **반드시** 넣는다. 기본값(262k)이면 KV 캐시가 80GB 를
다 먹는다.

카드가 여러 장이면 나눈다 — vLLM 은 0번, OCR 은 `--gpu-id 1`.

### 4. 실행

```bash
python scripts/run.py 문서.png -o out/
python scripts/run.py 계약서.pdf -o out/ --pages 1-3,7
python scripts/run.py data/*.png data/*.pdf -o out/
```

`out/*.boxes.png` 를 열어 **좌표가 맞는지 눈으로 확인**하는 게 가장 빠르다.
초록은 OCR 이 확정한 좌표, 빨강은 VLM 근사 좌표(검토 필수)다.

---

## 학습 — OCR 좌표를 VLM 에 이식

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 맞춰서
pip install -r requirements-train.txt
```

### ① 이미지 → 라벨 (사람 라벨 0건)

```bash
python scripts/build_grounding_data.py <문서들> -o data/g -c config/default.yaml --overlay 20
```

문서는 **아무거나** 된다. 개인정보가 없어도 좋다 — 가르치는 것은 "글자가
어디 있는가" 뿐이다. `data/g/overlay/*.png` 를 **눈으로 확인**하고 넘어간다.
박스가 글자에 안 붙어 있으면 여기서 멈춘다.

| 학습 샘플 | |
|---|---|
| **입력** | 타일 이미지 (추론에서 모델이 받는 것과 동일) |
| **정답** | `{"findings":[{"text":"강동혁","bbox_2d":[318,266,398,317]}, ...]}` |
| 손실 | 정답 토큰만. 프롬프트는 마스킹 |

`bbox_2d` 는 OCR 픽셀 박스를 추론 디코딩의 **역함수**로 환산한 값이라, 모델이
정답대로 뱉으면 픽셀 좌표가 그대로 복원된다 (왕복 오차 0px, 빌더가 assert 로 검산).

### ② 학습

```bash
python scripts/train_grounding.py --data data/g/data.jsonl --dry-run          # 배선 확인
python scripts/train_grounding.py --data data/g/data.jsonl -o out/s --max-samples 8   # 스모크
python scripts/train_grounding.py --data data/g/data.jsonl -o out/lora --merge out/merged
```

LoRA 는 LLM 에, **vision merger 는 전체 학습**이 기본값이다. 32px 토큰 격자
아래의 정밀도가 merger 에서 결정되므로 여기를 닫으면 격자 아래로 못 내려간다
(`--no-merger` 로 A/B).

`--merge` 로 저장한 가중치를 vLLM 에 올린다.

### ③ 평가 — **반드시 두 축을 함께**

```bash
python scripts/diagnose.py <평가 페이지들>    # 좌표
python scripts/measure.py  <평가 페이지들>    # 재현율
```

**재현율이 떨어졌으면 좌표가 좋아졌어도 실패다.** 개인정보 판단 능력이 이
프로젝트의 자산이고, 좌표 학습으로 그걸 잃으면 손해다.

> 학습된 모델을 올릴 때 `llm/prompts.py` 의 `_BBOX_PRECISION_COARSE` 를
> `_BBOX_PRECISION_TIGHT` 로 바꿔야 한다. 정밀하게 학습시켜 놓고 추론에서
> "대충 잡아라" 라고 지시하면 학습한 것을 되돌린다.

---

## 출력

```json
{
  "page": {"width": 1748, "height": 2480},
  "regions": [{
    "id": "r001", "type": "RRN", "bbox": [412, 780, 690, 812],
    "source": "ocr_refined", "confidence": 1.0,
    "text": "901231-1234563", "vlm_text": "901231-1234563",
    "field": "본인 주민등록번호", "char_span": [0, 14],
    "verified": true, "checksum": "ok", "agreement": "exact",
    "coarse": false, "needs_review": false,
    "reason": "크롭 OCR 과 완전일치; 체크섬 통과"
  }],
  "stats": {"n_findings": 24, "localized_rate": 0.875, "n_needs_review": 4}
}
```

| 필드 | 뜻 |
|---|---|
| `bbox` | **최종 좌표.** 마스킹은 이것만 쓰면 된다 |
| `source` | `ocr_refined`(정확) / `vlm_coarse`(근사, 검토 필수) |
| `text` / `vlm_text` | OCR 이 읽은 값 / VLM 이 읽은 값. 어긋나면 어느 엔진 문제인지 보인다 |
| `agreement` | 두 엔진 교차검증. `exact` 면 값이 거의 확실하다 |
| `checksum` | `ok` / `failed` / `null`(그 라벨엔 체크섬 없음) |
| `char_span` | 부분 마스킹(`901231-1******`)용 문자 오프셋 |
| `needs_review` | 사람이 봐야 한다 |

`--include-ocr` 를 주면 VLM 원본 판단과 OCR 박스, 원시응답까지 들어간다 (**진단 필수**).

**마스킹 방침**: `coarse` 나 `needs_review` 를 **가리지 않는 쪽으로 처리하면 안 된다.**
미탐(개인정보 노출)이 과탐(멀쩡한 칸 덧칠)보다 훨씬 비싸다.

---

## 자주 만지는 설정

`config/default.yaml` (전체 주석 있음). 우선순위: 코드 기본값 → 파일 → `PII_*` → CLI.

| 키 | 기본 | 언제 |
|---|---|---|
| `detect.tiles` | 3 | 표가 빽빽해 뒷부분이 잘리면 4~5 |
| `detect.samples` | 1 | **학습 없이 미탐을 줄이는 유일한 손잡이.** 호출이 `tiles × samples` 로 는다 |
| `detect.coord_convention` | auto | `diagnose.py` 로 확인 후 `per-mille` 고정 권장 |
| `locate.min_pad_px` | 64 | 크롭 여유. 모델 grounding 한계(32px)의 2배 |
| `locate.upscale` | 2.0 | 작은 글씨 인식률 |
| `ocr.gpu_id` | 0 | vLLM 과 카드를 나눌 때 |
| `pipeline.target_long_side` | 2464 | **`image_factor`(32)의 배수로** |

---

## 문제가 생기면

| 증상 | 도구 |
|---|---|
| 박스가 밀린다 | `scripts/diagnose.py` — 규약 오류 / 체계적 밀림 / 무작위 오차로 판정. **문서 내용을 출력하지 않는다** |
| 미탐이 있다 | `scripts/measure.py` — 설정을 바꿔가며 비교. 사람 라벨 불필요 |
| 어느 항목이 왜 그런지 | `--include-ocr` 로 뽑아 `regions[].reason` 확인 |

---

## 개인정보 라벨 (18종)

`NAME` `RRN` `ADDRESS` `EMAIL` `IP` `ACCOUNT_NO` `CARD_NO` `PHONE` `PASSPORT`
`FOREIGN_ID` `DRIVER_LICENSE` `BIZ_NO` `CORP_NO` `BIRTH` `ORG` `TITLE`
`SIGNATURE` `OTHER`

탐지는 전부 VLM 이 한다. 체크섬이 있는 라벨(`RRN` `BIZ_NO` `CARD_NO` 등)만
`rules/checksums.py` 가 **검증**한다 — 규칙은 탐지기가 아니라 검증기다.
라벨을 바꾸려면 `schema.py` 의 `PII_LABELS` 만 고치면 프롬프트와 스키마가 따라온다.

---

## 구조

```
src/pii_pipeline/
  pipeline.py      전처리 → VLM 탐지 → 좌표 확정 → 검증
  detect.py        ① VLM 이 개인정보를 전사한다 (타일 분할, 좌표 규약)
  locate.py        ② 크롭 OCR 로 좌표를 확정한다  ← 학습 후 축소 대상
  verify.py        ③ 체크섬·자리수·항목명 필터
  normalize.py     회피 표기 접기 ("공1공-1234" → "010-1234")
  train/           학습 데이터 구축 (추론에 관여하지 않는다)
scripts/
  run.py                     실행
  build_grounding_data.py    학습 데이터 생성
  train_grounding.py         LoRA 학습
  diagnose.py / measure.py   진단 / 비교
  make_synthetic.py          정답 딸린 합성 서식
```

설계 근거와 과거에 버린 구조는 [`docs/DESIGN.md`](docs/DESIGN.md) 에 있다.
