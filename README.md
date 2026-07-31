# 금융 문서 개인정보 영역 탐지 파이프라인

이미지를 입력받아 **개인정보 영역의 좌표와 유형**을 반환한다.
마스킹/치환은 이 파이프라인의 출력을 소비하는 **별도 모듈**에서 수행한다.

```
입력: 금융 문서 이미지 1장
출력: [{"bbox": [412, 780, 690, 812], "type": "RRN", ...}, ...]
```

현재 용도는 **학습데이터 구축**이다. 따라서 정책은 **recall 최우선**이며,
과검을 줄이는 제외 규칙은 넣지 않았다 ([확장 포인트](#제외-규칙-추가하기) 참조).

---

## 파이프라인

```
금융 이미지 (1장)
   │
   ├─① 전처리        기울기 보정 + 해상도 정규화
   │
   ├─② PaddleOCR     det/rec 분리. 읽지 못한 박스도 번호를 부여해 유지
   │                 → [12] (0.31,0.51) 홍길동
   │                 → [13] (0.31,0.58) ???  <OCR_FAILED>
   │
   ├─③ 규칙 레이어    정규식 + 체크섬으로 정형 항목 확정
   │                 주민번호·외국인번호·법인번호·사업자번호·카드·계좌·전화·이메일
   │
   ├─④ LLM pass-1    ③이 못 잡은 문맥 항목만 판단 (텍스트만 입력)
   │                 이름·주소·소속·직위 등
   │                 → {"idx":[12], "type":"NAME"}
   │
   ├─④' VLM pass-2   이미지를 직접 보고 누락 회수
   │                 손글씨·도장·서명·OCR 실패 박스
   │                 → {"idx":[13], "type":"NAME", "reason":"손글씨 성명"}
   │
   ├─⑤ 병합·검증      범위/중복/공간 검사 + 제외 훅
   │
   └─⑥ 결과 JSON     bbox + type + source 티어
```

## 핵심 설계 결정 4개

### 1. 좌표는 OCR 에서, 판단만 LLM 이 한다

LLM 에 좌표를 물으면 **환각한다.** 금융 서식처럼 셀이 촘촘한 문서에서는 옆 칸을
가리킨다. 그래서 OCR 박스에 번호를 매겨 넘기고 **번호만 회수**한다.
좌표 정확도는 OCR 수준(픽셀 단위)으로 보장된다.

`text` 역시 LLM 출력을 쓰지 않고 **OCR 원문에서 재조립**한다.
텍스트 환각이 구조적으로 불가능해진다.

### 2. 규칙이 LLM 보다 먼저다

주민번호·사업자번호·카드번호는 체크섬으로 거의 100% 잡힌다.
**가장 중요한 항목을 모델 운에 맡기지 않는다.** LLM 은 규칙으로 못 잡는
문맥 의존 항목(이름/주소/소속)만 담당한다.

### 3. OCR det/rec 을 분리한다

미탐에는 두 종류가 있다.

| 유형 | 원인 | 텍스트 pass | 이미지 pass |
|---|---|---|---|
| **분류 누락** | OCR 은 읽었는데 개인정보로 판단 안 함 | 잡힘 | 잡힘 |
| **OCR 누락** | 손글씨·도장·서명 → 애초에 못 읽음 | **구조적으로 불가** | 잡힘 |

두 번째가 훨씬 위험하다. PaddleOCR 의 **검출기는 인식기보다 강해서** 손글씨나
도장에 겹친 글자도 "여기 뭔가 있다"는 건 잡아낸다. 그래서 **rec 실패 박스를
버리지 않고 번호를 부여해 유지**한다. 그러면 이미지 pass 도 좌표를 발명할 필요
없이 번호로 답할 수 있다.

det 조차 놓친 잔여분만 VLM 좌표를 쓰고, `coarse` + `needs_review` 로 태깅한다.

### 4. 주소는 박스 여러 개 → 합쳐서 반환

OCR 은 주소를 2~4조각으로 쪼갠다. LLM 이 인덱스 **그룹**으로 답하고,
후처리에서 합집합 bbox 를 계산한다. `member_boxes` 를 함께 남기므로
다운스트림이 통짜/조각별 마스킹을 선택할 수 있다.

그룹은 **실제 픽셀 간격**으로 재검증한다 (행 번호 차이는 물리적 거리를
반영하지 못한다 — 행 2칸 차이가 600px 일 수 있다).

---

## 설치

```bash
pip install -r requirements.txt
pip install -e .
```

`paddlepaddle-gpu` 는 CUDA 버전에 맞는 휠을 설치해야 한다.
자세한 내용은 `requirements.txt` 주석 참조.

### 폐쇄망 반입 (중요)

**PaddleOCR 은 첫 실행 시 모델을 자동 다운로드한다.** 폐쇄망에서는 여기서
실패하므로 반드시 사전 반입이 필요하다.

```bash
# ① 외부망 장비에서
python scripts/download_models.py --out ./ocr_models
tar czf ocr_models.tar.gz ocr_models/

# ② 폐쇄망에 반입 후
tar xzf ocr_models.tar.gz
export PII_OCR_DET_DIR=$PWD/ocr_models/det
export PII_OCR_REC_DIR=$PWD/ocr_models/rec
export PII_OCR_CLS_DIR=$PWD/ocr_models/cls
```

모델 가중치(Qwen3.5-9B)도 함께 반입해야 한다.

---

## vLLM 서버 기동

```bash
vllm serve Qwen/Qwen3.5-9B \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --limit-mm-per-prompt image=1 \
  --enable-prefix-caching \
  --guided-decoding-backend xgrammar
```

### 꼭 지켜야 할 설정

| 설정 | 이유 |
|---|---|
| `--max-model-len 8192` | 기본값(262k)이면 KV 캐시가 80GB 를 다 먹는다. 1페이지는 3k 토큰으로 충분하다. |
| `--dtype bfloat16` | 9B 는 A100 80GB 에 여유롭게 올라간다. 양자화로 정확도를 깎을 이유가 없다. |
| `--gpu-memory-utilization 0.80` | PaddleOCR 몫을 남긴다 (`OcrConfig.gpu_mem` 과 짝을 맞춘다). |
| `enable_thinking=False` | 코드에서 항상 끈다. Qwen3 계열은 추론 모드가 켜지면 사고 토큰을 수천 개 뱉어 지연시간 예산을 날린다. |
| `temperature=0` | 결정론적 출력. 학습데이터 구축과 감사 대응에 필수. |

---

## 실행

```bash
# 검증용 합성 데이터 생성 (정답 JSON 포함)
python scripts/make_synthetic.py --out data/synth --count 5

# 처리 + 시각화 검수 이미지
python scripts/run.py data/synth/*.png -o out/ --overlay

# 텍스트 pass 만 (베이스라인 측정)
python scripts/run.py sample.png -o out/ --no-pass2

# blind 모드 (앵커링 편향 비교)
python scripts/run.py sample.png -o out/ --pass2-blind
```

라이브러리로 쓸 때:

```python
from pii_pipeline import PiiPipeline, PipelineConfig

pipeline = PiiPipeline(PipelineConfig(enable_pass2=True))
result = pipeline.run("document.png")

for region in result.regions:
    print(region.type, region.bbox, region.source.value, region.confidence)
```

> OCR 엔진과 LLM 클라이언트는 인스턴스 수명 동안 재사용된다.
> 배치 처리 시 `PiiPipeline` 인스턴스를 **하나만** 만들 것.

---

## 출력 계약

```json
{
  "image_path": "document.png",
  "page": {"width": 1748, "height": 2480},
  "regions": [
    {
      "id": "r001",
      "type": "RRN",
      "bbox": [412, 780, 690, 812],
      "source": "rule",
      "confidence": 1.0,
      "text": "901231-1234563",
      "member_boxes": [[414, 782, 688, 810]],
      "member_index": [7],
      "char_span": [0, 14],
      "ocr_status": "ok",
      "coarse": false,
      "needs_review": false,
      "low_confidence": false,
      "reason": "체크섬 통과"
    },
    {
      "id": "r002",
      "type": "ADDRESS",
      "bbox": [412, 850, 1180, 920],
      "source": "llm_pass1",
      "confidence": 0.93,
      "text": "서울특별시 강남구 테헤란로 123 ○○빌딩 5층",
      "member_boxes": [[412,850,660,882],[670,850,900,882],[412,888,1180,920]],
      "member_index": [12, 13, 14],
      "ocr_status": "ok",
      "needs_review": false
    },
    {
      "id": "r003",
      "type": "SIGNATURE",
      "bbox": [1228, 1998, 1432, 2074],
      "source": "vlm_grounding",
      "confidence": 0.82,
      "text": null,
      "member_index": [],
      "ocr_status": "failed",
      "coarse": true,
      "needs_review": true,
      "reason": "자필 서명"
    }
  ],
  "timings": {"preprocess": 0.31, "ocr": 1.22, "rules": 0.01,
              "llm_pass1": 1.84, "llm_pass2": 2.41, "merge": 0.02, "total": 5.81},
  "warnings": [],
  "stats": {"n_ocr_boxes": 47, "n_regions": 12, "n_needs_review": 3,
            "by_source": {"rule": 5, "llm_pass1": 4, "vlm_pass2": 2, "vlm_grounding": 1},
            "by_type": {"RRN": 1, "NAME": 3, "ADDRESS": 1, "...": 0}}
}
```

### `source` 티어 — 다운스트림이 정책을 계층화할 수 있다

| source | 의미 | 권장 처리 |
|---|---|---|
| `rule` | 체크섬 통과 | 자동 처리. 검토 불필요. |
| `llm_pass1` | OCR 텍스트 기반 분류 | 자동 처리. |
| `vlm_pass2` | 이미지 pass 회수. **OCR 좌표** 사용 | 자동 처리 + 검토 큐 |
| `vlm_grounding` | VLM 좌표 근사. 부정확 | 검토 필수 |

`id` 는 읽기 순서 기반으로 **결정론적**으로 부여된다. 같은 입력을 재처리하면
같은 ID 가 나오므로 어노테이션 재연결이 가능하다.

`char_span` 은 규칙 레이어 탐지 시 박스 텍스트 내 문자 오프셋이다.
부분 마스킹(`901231-1******`)을 구현할 다운스트림이 쓴다.
단 `bbox` 는 항상 **박스 전체 영역**임에 유의 — OCR 이 문자 단위 좌표를
주지 않으므로 글자수 비례 분할은 근사치다.

---

## 라벨 스키마

| 규칙 레이어가 확정 (체크섬/패턴) | LLM 이 판단 (문맥) |
|---|---|
| `RRN` `FOREIGN_ID` `PASSPORT` `DRIVER_LICENSE` `BIZ_NO` `CORP_NO` `ACCOUNT_NO` `CARD_NO` `PHONE` `EMAIL` | `NAME` `ADDRESS` `BIRTH` `ORG` `TITLE` `SIGNATURE` `OTHER` |

`src/pii_pipeline/schema.py` 의 `PII_LABELS` / `RULE_LABELS` 에서 수정한다.
프롬프트와 guided-decoding 스키마가 이 상수에서 자동 생성되므로 한 곳만 고치면 된다.

---

## 예상 지연시간

A4 1장, A100 80GB, Qwen3.5-9B bf16 기준:

| 단계 | 시간 |
|---|---|
| 전처리 | 0.2~0.5s |
| PaddleOCR (det+rec) | 0.5~1.5s |
| 규칙 레이어 | <0.05s |
| LLM pass-1 (출력 ~120tok) | 1.5~2s |
| VLM pass-2 (이미지 ~2.5k tok) | 2~3s |
| 병합·검증 | <0.1s |
| **합계** | **약 5~8초** |

30초 목표 대비 4~5배 여유가 있다. 실측값은 결과 JSON 의 `timings` 에 기록된다.

`--image-max-side` 를 너무 줄이면 모델이 작은 글씨를 못 읽어 pass-2 의 존재
의미가 사라진다. **1500~2000px 는 유지할 것.**

---

## 확장 포인트

### 제외 규칙 추가하기

`src/pii_pipeline/exclusions.py` 의 `should_exclude()` 는 **현재 아무것도
제외하지 않는다.** 운영 단계에서 과검을 줄여야 할 때 이 파일만 고치면 된다
(파일 안에 예시가 주석으로 들어있다).

`Source.RULE` 항목은 체크섬을 통과한 확정 건이므로 제외 대상에서 빼두는 것을 권한다.

### pass-2 앵커링

1차 결과를 보여주면 모델이 "잘 됐네"로 동조하는 편향이 생길 수 있다.
`PipelineConfig(pass2_blind=True)` 로 1차 결과를 감춘 독립 판단과 비교해볼 수 있다.
검증셋에서 두 방식의 recall 을 재보고 결정할 것.

### 검증 방법

`scripts/make_synthetic.py` 는 **정답 JSON 을 함께 출력**한다.
의도적으로 손글씨/도장/저대비/다단 컬럼을 섞어두었으므로 난이도별 recall 을
분리해 측정할 수 있다 (`difficulty` 필드).

시각화 검수는 `--overlay` 가 가장 빠르다. 색상은 `source` 티어별로 구분된다.

---

## 좌표계 주의

결과 좌표는 **전처리된 이미지 기준**이다. 원본 좌표계로 되돌려야 하면
`PreprocessResult.transform`(scale / rotation_deg / rotation_center)으로 역변환한다.
기울기 보정은 각도가 0.4°~15° 구간일 때만 적용된다.

## 테스트

```bash
python -m pytest
```

OCR/LLM 의존성 없이 순수 로직(체크섬·정규식·정렬·병합·프롬프트·합성기)을 검증한다.

## 주의

- 이 저장소의 모든 번호는 **체크섬 알고리즘으로 합성한 가짜 값**이다.
- 실제 내부 자료를 저장소에 커밋하지 말 것. 검증은 합성 데이터로 한다.
