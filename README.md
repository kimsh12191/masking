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
python -m pytest          # 877 passed
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
  --guided-decoding-backend xgrammar \
  --trust-remote-code
```

`--trust-remote-code` 를 빼면 `Failed to load the tokenizer` 로 죽는다 —
Qwen-VL 의 토크나이저·프로세서가 transformers 에 내장되지 않은 커스텀 코드다.

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

> **설정은 추론과 같은 파일에 있다** — `config/default.yaml` 의 `grounding:`
> (데이터 생성) 과 `train:` (LoRA) 섹션. 전체 주석이 붙어 있고, 아래 CLI 플래그는
> 그 값을 **일회성으로 덮는 용도**다.
>
> 같은 파일에 둔 이유는 결합 때문이다. 위쪽 `pipeline.canvas` 와 `detect.tiles` 가
> **학습과 추론 양쪽의 기하를 동시에 정한다.** 따로 두면 갈리기 쉽고, 갈리면
> 틀린 좌표를 학습시키는데 에러도 안 난다.

### ① 이미지 → 라벨 (사람 라벨 0건)

```bash
python scripts/build_grounding_data.py <문서들> -o data/g -c config/default.yaml --overlay 20
```

문서는 **아무거나** 된다. 개인정보가 없어도 좋다 — 가르치는 것은 "글자가
어디 있는가" 뿐이다. `data/g/overlay/*.png` 를 **눈으로 확인**하고 넘어간다.
박스가 글자에 안 붙어 있으면 여기서 멈춘다.

**과제는 "값을 줄 테니 위치만 찍어라" 다.** 전사는 이미 잘하므로 안 가르친다 —
손실이 거의 전부 좌표에 걸리고, OCR 오독이 정답에 섞일 길도 없다.

```
[입력]  타일 이미지 1760×960  +

  다음 값들이 이 이미지에 있다. 각각의 위치를 **모두** 답하라.
  - 강동혁
  - 780914-2184953

[정답]
  {"findings":[{"text":"강동혁","bbox_2d":[318,266,398,317]},
               {"text":"강동혁","bbox_2d":[600,800,700,850]},   ← 같은 값 2곳
               {"text":"780914-2184953","bbox_2d":[670,667,886,719]}]}
```

손실은 정답 토큰에만 걸린다 (프롬프트는 마스킹).

| 설계 | 왜 |
|---|---|
| 질문은 **중복 제거**, 답은 **모든 출현** | 같은 이름·날짜가 여러 칸에 나오는 게 정상이다. 한 곳만 답하면 나머지가 안 가려진다 |
| **전부 묻지 않는다** (`--query-ratio 0.25,1.0`) | 추론에서는 텍스트 수십 줄 중 개인정보 몇 개만 답한다. 항상 전부 물으면 "보이는 것 다 답한다" 를 배운다 |
| 질문 순서를 섞는다 | 답은 읽기 순서다. 질문 순서대로 답하는 걸 배우면 추론에서 무너진다 |
| 입력 크기가 **항상 같다** | 페이지가 고정 캔버스(1760×2464), 타일 3개가 전부 1760×960 |
| `--aug-scales 1.5,2.0` | 더 작은 영역을 잘라 타일 크기로 확대. **크기는 그대로, 글자만 커 보인다.** 문서마다 폰트가 다르므로 한 스케일만 배우면 흔들린다 |

`bbox_2d` 는 OCR 픽셀 박스를 추론 디코딩의 **역함수**로 환산한 값이라, 모델이
정답대로 뱉으면 픽셀 좌표가 그대로 복원된다 (왕복 오차 0px, 빌더가 assert 로 검산).

> 도장·서명은 지목할 텍스트가 없어 이 과제로 못 가르친다. `--read-ratio` 로
> "전부 읽어라" 과제를 섞을 수 있지만 **기본은 꺼져 있다** — 좌표 능력은 내용과
> 무관해서 글자로 배운 것이 도장에도 쓰인다. 학습 후 실제로 나쁘면 그때 켠다.

### ② 학습

**평범한 SFT 다.** 좌표 회귀 헤드도, 특별한 손실도 없다.

```
[system] + [타일 이미지] + [user]  →  모델이 정답 JSON 을 생성
                                       ↑ 이 토큰들에만 cross-entropy
```

모델은 **이미 이 형식으로 답한다.** 숫자만 틀리니까 그 숫자를 정답으로 놓고
next-token 예측을 돌리는 것이 전부다. 프롬프트 토큰은 손실에서 뺀다 — 안 빼면
모델이 자기 지시문을 외운다.

```bash
python scripts/train_grounding.py --data data/g/data.jsonl --dry-run          # ① 배선 확인
python scripts/train_grounding.py --data data/g/data.jsonl -o out/s --max-samples 8   # ② 스모크
python scripts/train_grounding.py --data data/g/data.jsonl -o out/lora --merge out/merged
```

**①을 건너뛰지 마라.** GPU 없이 돌고, 손실이 걸리는 문자열을 디코딩해 보여준다.
거기 정답 JSON 만 나와야 한다. 지시문이 섞여 있거나 시작이 한 토큰 밀리면
**학습 로그에는 아무 징후가 없고** "좌표가 안 좋아지네" 로만 나타난다.

**어디를 여는가**

| 부위 | 기본 | 왜 |
|---|---|---|
| LLM attention/MLP | LoRA r=16 | |
| **vision merger** | **전체 학습** | 패치 4개를 토큰 1개로 압축하는 자리. **32px 격자 아래 위치정보가 여기서 살아남느냐로 결정된다.** 선형층 두어 개라 통째로 열어도 싸다 |
| ViT | 동결 | `--vision-blocks N` 으로 상위 N개 블록에 LoRA |

ViT 를 기본으로 안 여는 건 **순서** 때문이다. merger 만 열고 먼저 재야 어디까지가
merger 몫인지 안다 (`--no-merger` 로 A/B). 격자 아래로 못 내려가면 그때 켠다.

**규모**

| | |
|---|---|
| 하드웨어 | A100·H100 **1장** (9B bf16 + LoRA + grad checkpointing) |
| 시간 | 수 시간 |
| 데이터 | 페이지 500~2,000장이면 시작할 만하다 (장당 샘플 5개, 좌표 라벨 ~80개) |
| 기본값 | lr 1e-4, batch 1 × accum 8, 1 epoch, bf16 |

**서빙**은 `--merge` 로 베이스에 합친 체크포인트를 vLLM 에 올린다. LoRA 핫스왑은
비전타워 쪽 보장이 애매한데, 어차피 모델 하나만 쓰므로 머지가 확실하다.

### ③ 평가 — **반드시 두 축을 함께**

```bash
python scripts/diagnose.py <평가 페이지들>    # 좌표
python scripts/measure.py  <평가 페이지들>    # 재현율
```

**재현율이 떨어졌으면 좌표가 좋아졌어도 실패다.** 개인정보 판단 능력이 이
프로젝트의 자산이고, 좌표 학습으로 그걸 잃으면 손해다.

---

## 반드시 맞춰야 하는 것

### 추론 — 안 맞으면 안 돌거나 조용히 나빠진다

| 설정 | 왜 |
|---|---|
| `llm.model` / `llm.base_url` | vLLM 이 서빙하는 것과 **정확히** 같아야 한다 |
| `ocr.det_model_dir` / `rec_model_dir` / `cls_model_dir` | **폐쇄망 필수.** 없으면 첫 실행에서 자동 다운로드를 시도하다 죽는다 |
| `detect.image_factor` | **모델을 바꾸면 반드시 같이.** Qwen3-VL `32` / Qwen2·2.5-VL `28`. 틀리면 좌표가 위치에 비례해 밀리는데, 에러는 안 난다 |
| `pipeline.canvas` · `llm.image_max_side` | `image_factor` 의 **배수**. 아니면 서버가 조각을 다시 리샘플해 작은 글씨가 뭉개진다 = 미탐 |
| `ocr.gpu_id` | vLLM 과 같은 카드면 메모리 경합. 카드가 여러 장이면 나눈다 |

```bash
python scripts/run.py --print-config     # 위 조합을 검산해준다
```

### 학습 — 추론과 **같아야** 한다

| | 왜 |
|---|---|
| `build_grounding_data.py -c <서빙과 같은 config>` | **가장 중요.** 캔버스·타일이 다르면 **틀린 좌표를 학습시킨다.** 에러도 안 나고 결과만 나빠진다 |
| `train_grounding.py --model <서빙할 베이스 모델>` | 다르면 어댑터가 안 맞는다 |
| `--merger-module` | 못 찾으면 죽으면서 알려준다. `--list-modules` 로 실제 이름 확인 |

### 학습 후에는 바꾸지 마라

```
pipeline.canvas   detect.tiles   detect.overlap   detect.image_factor   llm.image_max_side
```

모델이 그 기하로 좌표를 배웠다. 바꾸면 **본 적 없는 입력**이 되므로 `diagnose.py`
로 다시 재야 한다. 학습 전에 확정해 두는 것이 안전하다.

그리고 학습된 모델을 올릴 때 **프롬프트 한 곳**을 바꾼다 —
`llm/prompts.py` 의 `_BBOX_PRECISION_COARSE` → `_BBOX_PRECISION_TIGHT`.
정밀하게 학습시켜 놓고 추론에서 "대충 잡아라" 라고 지시하면 학습한 것을 되돌린다.

---

## 출력

```json
{
  "page": {"width": 1760, "height": 2464},
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
| `pipeline.canvas` | [1760, 2464] | **고정 캔버스.** 모든 페이지가 이 크기. `image_factor`(32)의 배수로 |
| `grounding.*` | | 학습 데이터 생성 (추론에서는 안 읽는다) |
| `train.*` | | LoRA 학습 (추론에서는 안 읽는다) |

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
