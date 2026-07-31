# 개인정보 영역 탐지 파이프라인

금융 문서에서 **개인정보 영역의 좌표와 유형**을 찾는다.
마스킹/치환은 이 출력을 받는 별도 모듈에서 한다.

입력은 **이미지 또는 PDF**, 출력은 페이지당 두 파일이다.

```
이미지 1장   →  {이름}.boxes.png        {이름}.json
PDF          →  {이름}_p001.boxes.png   {이름}_p001.json
                {이름}_p002.boxes.png   {이름}_p002.json   ...
```

| 파일 | 내용 |
|---|---|
| `.boxes.png` | 박스 영역이 표시된 이미지 (눈으로 검수) |
| `.json` | 박스 위치 + 개인정보 유형 (다운스트림이 소비) |

두 파일의 좌표계는 같다. `boxes.png` 위의 박스와 `json` 의 `bbox` 가 1:1로 대응한다.

구성: `PaddleOCR` + `Qwen3.5-9B` (A100 80GB 1장, 1페이지 5~8초 예상)

상세 설계와 근거는 [`docs/DESIGN.md`](docs/DESIGN.md) 참조.

---

## 1. 지금 바로 테스트 (GPU/모델 없이)

로직 검증은 의존성 없이 돌아간다.

```bash
git clone <repo> && cd masking
git checkout claude/personal-info-masking-pipeline-erken9

pip install pytest Pillow PyYAML numpy pypdfium2
python -m pytest
```

→ **610건 통과**하면 체크섬·정규식·읽기순서 정렬·병합 검증·프롬프트·설정 로딩·
라벨 스키마·PDF 페이지 분해·파이프라인 배선이 모두 정상이다.

설정만 확인해보려면:

```bash
python scripts/run.py --print-config
```

합성 서식 이미지도 이 상태에서 만들어볼 수 있다 (정답 JSON 포함).

```bash
python scripts/make_synthetic.py --out data/synth --count 3
```

`data/synth/synth_001.png` 을 열어보면 손글씨·도장·저대비 글씨가 섞인
가짜 여신거래약정서가 나온다. `synth_001.truth.json` 이 정답이다.

---

## 2. 전체 파이프라인 실행 (GPU 필요)

### 설치

```bash
pip install -r requirements.txt
pip install -e .
```

`paddlepaddle-gpu` 는 CUDA 버전에 맞는 휠을 따로 설치해야 한다
(`requirements.txt` 주석 참조).

### 폐쇄망이면 — 모델 먼저 반입

PaddleOCR 은 첫 실행 시 모델을 자동 다운로드한다. 폐쇄망에서는 여기서 죽는다.

```bash
# 외부망 장비에서
python scripts/download_models.py --out ./ocr_models
tar czf ocr_models.tar.gz ocr_models/

# 폐쇄망에 반입 후
tar xzf ocr_models.tar.gz
export PII_OCR_DET_DIR=$PWD/ocr_models/det
export PII_OCR_REC_DIR=$PWD/ocr_models/rec
export PII_OCR_CLS_DIR=$PWD/ocr_models/cls
```

Qwen3.5-9B 가중치도 함께 반입해야 한다.

### vLLM 기동

```bash
vllm serve Qwen/Qwen3.5-9B \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt image=1 \
  --enable-prefix-caching \
  --guided-decoding-backend xgrammar
```

`--max-model-len 8192` 를 **반드시** 넣어야 한다. 기본값(262k)이면 KV 캐시가
80GB 를 다 먹는다.

### 실행

```bash
# 이미지
python scripts/run.py data/synth/*.png -o out/

# PDF — 페이지별로 나온다
python scripts/run.py 계약서.pdf -o out/

# PDF 페이지 범위만
python scripts/run.py 계약서.pdf -o out/ --pages 1-3,7

# 이미지와 PDF 를 섞어도 된다
python scripts/run.py data/*.png data/*.pdf -o out/
```

```
out/
├── synth_001.boxes.png      ← 이미지 입력
├── synth_001.json
├── 계약서_p001.boxes.png     ← PDF 입력 (페이지별)
├── 계약서_p001.json
├── 계약서_p002.boxes.png
└── 계약서_p002.json
```

`boxes.png` 를 열어 **좌표가 맞는지 눈으로 확인**하는 게 가장 빠르다.
라벨은 박스 **바깥**에 붙으므로 안의 내용을 가리지 않는다.

| 색 | 의미 |
|---|---|
| 초록 | 체크섬 확정 (주민번호·카드 등). 검토 불필요 |
| 파랑 | 텍스트 pass 분류 (이름·주소 등) |
| 주황 | 이미지 pass 에서 회수 (손글씨·도장). OCR 좌표 사용 |
| 빨강 | VLM 좌표 근사. 검토 필수 |
| `!` | `needs_review` 플래그 |

---

## 3. 권장 검증 순서

```bash
# ① 텍스트 pass 만 — 베이스라인
python scripts/run.py data/synth/*.png -o out_base/ --no-pass2

# ② 이미지 pass 켜고 — 회수량 비교
python scripts/run.py data/synth/*.png -o out_full/
```

②에서 손글씨·도장 항목이 주황색으로 잡히면 이미지 pass 가 값을 하는 것이다.
합성 데이터 정답의 `difficulty` 필드로 난이도별 recall 을 분리해 측정하면 된다.

| `difficulty` | 무엇을 검증하나 |
|---|---|
| `handwriting` | OCR rec 실패 → 이미지 pass 회수 |
| `stamp_overlap` | 도장에 가린 글자 |
| `signature` | 자필 서명 (OCR 박스가 없을 수도 있다) |
| `inline_label` | `담당자: 조민석` — 라벨과 값이 **한 박스**. 실제 문서에 흔하다 |
| `unlabeled_value` | 라벨 없는 `(1978-04-15)` — 형태와 문맥으로만 판단 |

실측 소요시간은 결과 JSON 의 `timings` 에 단계별로 기록된다.

---

## 4. 설정

모델명·엔드포인트·OCR 경로 등은 **`config/default.yaml`** 에 있다. 코드를 고칠 필요 없다.

```yaml
llm:
  model: Qwen/Qwen3.5-9B                 # vLLM 기동 시 지정한 것과 일치해야 함
  base_url: http://127.0.0.1:8000/v1
  enable_thinking: false                 # 켜면 지연시간 예산이 날아간다
  image_max_side: 1800                   # 1500 이상 유지

ocr:
  lang: korean
  det_model_dir: null                    # 폐쇄망에서는 필수
  rec_model_dir: null

pipeline:
  enable_pass2: true

output:
  out_dir: out
  write_image: true
```

적용 순서 — **뒤가 앞을 덮는다.**

```
코드 기본값  →  config.yaml  →  환경변수(PII_*)  →  CLI 플래그
```

```bash
# 현재 적용된 설정만 확인
python scripts/run.py --print-config

# 설정 파일 지정
python scripts/run.py sample.png --config config/prod.yaml

# 일회성 변경 (파일은 그대로)
python scripts/run.py sample.png --model /opt/models/qwen35-9b --no-pass2
```

파일 탐색 순서: `--config` > `$PII_CONFIG` > `./config.yaml` > `./config/default.yaml`

환경 고유값만 환경변수로 빼도 된다:

```bash
export PII_LLM_MODEL=/opt/models/qwen35-9b
export PII_LLM_BASE_URL=http://10.0.0.5:8000/v1
export PII_OCR_DET_DIR=/opt/ocr_models/det
export PII_OCR_REC_DIR=/opt/ocr_models/rec
```

> **오타난 설정 키는 조용히 무시되지 않고 실행이 즉시 실패한다.**
> 폐쇄망에서 설정이 반영되지 않은 채 도는 것이 가장 찾기 어려운 실패다.
>
> ```
> 설정 오류: 설정 섹션 'llm' 에 알 수 없는 키가 있습니다: modle
>   사용 가능한 키: api_key, base_url, enable_thinking, ..., model, ...
> ```

환경별로 다른 설정을 쓸 때는 루트에 `config.yaml` 을 두면 자동으로 잡힌다
(`.gitignore` 에 있으므로 커밋되지 않는다). `config/default.yaml` 은 템플릿이다.

---

## 5. 주요 옵션

```bash
python scripts/run.py --help
```

| 옵션 | 용도 |
|---|---|
| `--print-config` | 적용된 설정을 출력하고 종료 |
| `--config` | 설정 파일 경로 |
| `--no-image` | 박스 표시 이미지 생략, JSON 만 저장 |
| `--pages` | PDF 페이지 범위. `1-3,7` / `5-` |
| `--pdf-password` | 암호화된 PDF 암호 (`PII_PDF_PASSWORD` 환경변수 권장) |
| `--font` | 한글 폰트 경로 (없어도 라벨은 ASCII 라 표시됨) |
| `--show-ocr-boxes` | OCR 박스 전체를 회색으로 표시 (디버깅) |
| `--no-pass2` | 이미지 pass 끄기 (베이스라인) |
| `--pass2-blind` | 1차 결과를 감추고 독립 판단 (앵커링 편향 비교) |
| `--include-ocr` | JSON 에 OCR 박스 + LLM 원시응답 포함 (디버깅) |
| `--image-max-side` | pass2 이미지 크기. **1500 이상 유지** (작으면 작은 글씨를 못 읽음) |
| `--cpu` | OCR 을 CPU 로 |

라이브러리로:

```python
from pii_pipeline import PiiPipeline, load_config, save_result

config = load_config()                       # config.yaml + 환경변수 적용
pipeline = PiiPipeline(config.pipeline)      # 배치 처리 시 인스턴스는 하나만

result = pipeline.run("document.png")
save_result(result, config.output.out_dir)   # boxes.png + json 저장

for r in result.regions:
    print(r.type, r.bbox, r.source.value, r.confidence)
```

설정 파일 없이 코드로만 구성해도 된다:

```python
from pii_pipeline import PiiPipeline, PipelineConfig
from pii_pipeline.llm.client import LlmConfig

pipeline = PiiPipeline(PipelineConfig(
    llm=LlmConfig(model="/opt/models/qwen35-9b", base_url="http://10.0.0.5:8000/v1"),
))
```

PDF 와 이미지를 섞어서 (한 장씩 즉시 저장하고 이미지 메모리를 해제한다):

```python
# 확장자를 보고 알아서 처리한다. PDF 1개 -> 페이지별 결과 여러 개
results = pipeline.run_any("계약서.pdf", out_dir="out/", pages="1-3")

# 섞어서 일괄 처리 — 반환 개수가 입력 개수와 다를 수 있다
results = pipeline.run_batch(["a.png", "b.pdf"], out_dir="out/")
```

> PDF 페이지는 제너레이터로 한 장씩 렌더링한다. A4 300dpi 기준 장당 약 13MB
> 이므로 `out_dir` 없이 큰 문서를 돌리면 메모리를 많이 쓴다.

---

## 6. 출력 형식

`{이름}.json`:

```json
{
  "image_path": "계약서.pdf",
  "page_no": 2,
  "page": {"width": 1748, "height": 2480},
  "regions": [
    {
      "id": "r001",
      "type": "RRN",
      "bbox": [412, 780, 690, 812],
      "text": "901231-1234563",
      "source": "rule",
      "confidence": 1.0,
      "member_boxes": [[414, 782, 688, 810]],
      "needs_review": false,
      "coarse": false
    }
  ],
  "timings": {"ocr": 1.22, "llm_pass1": 1.84, "llm_pass2": 2.41, "total": 5.81},
  "stats": {"n_regions": 12, "n_needs_review": 3, "by_source": {"rule": 5}}
}
```

- `source` — `rule`(체크섬) / `llm_pass1` / `vlm_pass2` / `vlm_grounding`.
  다운스트림이 티어별로 정책을 다르게 줄 수 있다.
- `member_boxes` — 주소처럼 여러 박스에 걸친 항목의 구성 박스.
  통짜/조각별 마스킹을 선택할 수 있다.
- `id` — 읽기 순서 기반 결정론적 부여. 재처리 시 같은 ID → 어노테이션 재연결 가능.
- `page_no` — PDF 의 1-기반 페이지 번호. 단일 이미지 입력이면 `null`.
  `image_path` 는 PDF 파일 경로를 그대로 유지한다.

---

## 7. 개인정보 라벨 (18종)

### 핵심 9종

| 라벨 | 항목 | 탐지 방식 | 검증 |
|---|---|---|---|
| `NAME` | 이름 | LLM (문맥) | — |
| `RRN` | 주민등록번호 | 규칙 | **체크섬** |
| `ADDRESS` | 주소 | LLM (문맥) | — |
| `EMAIL` | 이메일 | 규칙 | 패턴 |
| `IP` | IP 주소 | 규칙 | 구조 (IPv4 옥텟 범위 / IPv6 그룹 수) |
| `ACCOUNT_NO` | 계좌번호 | 규칙 + **문맥 키워드** | 없음 → 항상 검토 대상 |
| `CARD_NO` | 카드번호 | 규칙 | **Luhn** |
| `PHONE` | 전화번호 | 규칙 | 패턴 |
| `PASSPORT` | 여권번호 | 규칙 | 패턴 |

### 추가 9종

| 라벨 | 항목 | 탐지 방식 | 검증 |
|---|---|---|---|
| `FOREIGN_ID` | 외국인등록번호 | 규칙 | **체크섬** (2020-10 이후 발급분은 체크섬 없음) |
| `DRIVER_LICENSE` | 운전면허번호 | 규칙 | 패턴 + 문맥 키워드 |
| `BIZ_NO` | 사업자등록번호 | 규칙 | **체크섬** |
| `CORP_NO` | 법인등록번호 | 규칙 | **체크섬** |
| `BIRTH` | 생년월일 | LLM (문맥) | — |
| `ORG` | 소속/직장명 | LLM (문맥) | — |
| `TITLE` | 직위/직책 | LLM (문맥) | — |
| `SIGNATURE` | 서명/인영 | LLM (문맥) | — |
| `OTHER` | 위에 없는 개인식별정보 | LLM (문맥) | — |

### 담당 분리

```
규칙 레이어 (11종)   RRN FOREIGN_ID CORP_NO BIZ_NO DRIVER_LICENSE
                     PASSPORT ACCOUNT_NO CARD_NO PHONE EMAIL IP
                     → 정규식 + 체크섬. LLM 이 다시 판단하지 않는다.

LLM 판단   (7종)     NAME ADDRESS BIRTH ORG TITLE SIGNATURE OTHER
                     → 문맥 의존 항목만 맡긴다.
```

`RRN` / `FOREIGN_ID` / `CORP_NO` 는 **모두 13자리라 형태가 같다.** 하나의 패턴으로
잡고 체크섬과 성별코드로 갈라낸다. 셋 다 실패하면 폐기하지 않고 `RRN` 후보로
남겨 검토 대상에 넣는다 (recall 우선).

`ACCOUNT_NO` 는 표준 체크섬이 없어서, 같은 행 근처에 문맥 키워드
(`계좌` `예금` `입금` `은행` 등)가 있을 때만 잡고 **항상 `needs_review`** 로 표시한다.

`IP` 는 IPv4(옥텟 0~255)와 IPv6(전체 8그룹 또는 `::` 압축형)를 따로 처리한다.
`12:34:56` 같은 시각 표기나 `2022. 10. 06.` 같은 날짜는 잡지 않는다.

### 라벨을 바꾸려면

`src/pii_pipeline/schema.py` 의 `PII_LABELS` / `RULE_LABELS` **한 곳만** 고치면 된다.
프롬프트와 guided-decoding 스키마가 이 상수에서 자동 생성된다.
정형 식별자를 추가할 때는 `rules/detectors.py` 에 패턴도 넣어야 하며,
`tests/test_schema.py` 가 누락을 잡아준다.

---

## 8. 아직 검증 안 된 부분

GPU 없는 환경에서 만들었으므로 다음 두 개는 **첫 실행 시 깨질 가능성이 있다.**

| 항목 | 확인할 것 |
|---|---|
| PaddleOCR 연동 | `paddle_runner.py` 의 `parse()` 가 실제 반환 형태와 맞는지. **2.x 기준으로 작성됨** — 3.x 설치하면 손봐야 한다 |
| vLLM 연동 | `guided_json` / `chat_template_kwargs` 가 실제로 먹는지 |

로직·배선은 테스트로 검증됐다. PDF 렌더링은 실제 pypdfium2 로 검증했다.

### 알려진 개선 여지 — 전자 PDF

전자적으로 생성된 PDF(스캔이 아닌)는 **텍스트 레이어에 정확한 좌표가 이미
들어 있다.** 그런 문서는 OCR 없이 좌표를 그대로 뽑는 게 완벽하고 훨씬 빠르다.
현재는 스캔/전자를 구분하지 않고 전부 이미지로 렌더링해 OCR 을 태운다.

금융 문서는 전자 생성 비중이 높으니 실측해보고 판단할 값이 있다.
`src/pii_pipeline/pdf.py` 에 같은 내용을 주석으로 남겨뒀다.

---

## 참고

- 저장소의 모든 번호는 체크섬으로 합성한 **가짜 값**이다.
- **실제 내부 자료를 커밋하지 말 것.** 검증은 합성 데이터로 한다.
  (`.gitignore` 에 `data/`, `samples/`, `*.pdf` 를 넣어뒀다.)
- 과검 제외 규칙은 `src/pii_pipeline/exclusions.py` 훅에 추가한다.
  현재는 recall 우선으로 아무것도 제외하지 않는다.
