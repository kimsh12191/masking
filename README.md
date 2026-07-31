# 개인정보 영역 탐지 파이프라인

금융 문서 이미지에서 **개인정보 영역의 좌표와 유형**을 찾아 JSON 으로 반환한다.
마스킹/치환은 이 출력을 받는 별도 모듈에서 한다.

```
입력: 이미지 1장  →  출력: [{"bbox": [412,780,690,812], "type": "RRN"}, ...]
```

구성: `PaddleOCR` + `Qwen3.5-9B` (A100 80GB 1장, 1페이지 5~8초 예상)

상세 설계와 근거는 [`docs/DESIGN.md`](docs/DESIGN.md) 참조.

---

## 1. 지금 바로 테스트 (GPU/모델 없이)

로직 검증은 의존성 없이 돌아간다.

```bash
git clone <repo> && cd masking
git checkout claude/personal-info-masking-pipeline-erken9

pip install pytest Pillow
python -m pytest
```

→ **313건 통과**하면 체크섬·정규식·읽기순서 정렬·병합 검증·프롬프트·파이프라인
배선이 모두 정상이다.

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
# 처리 + 검수 이미지 생성
python scripts/run.py data/synth/*.png -o out/ --overlay --show-ocr-boxes
```

`out/synth_001.overlay.png` 를 열어 **좌표가 맞는지 눈으로 확인**하는 게 가장 빠르다.

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
python scripts/run.py data/synth/*.png -o out_full/ --overlay
```

②에서 손글씨·도장 항목이 주황색으로 잡히면 이미지 pass 가 값을 하는 것이다.
합성 데이터 정답의 `difficulty` 필드(`handwriting` / `stamp_overlap` / `signature`)로
난이도별 recall 을 분리해 측정하면 된다.

실측 소요시간은 결과 JSON 의 `timings` 에 단계별로 기록된다.

---

## 4. 주요 옵션

```bash
python scripts/run.py --help
```

| 옵션 | 용도 |
|---|---|
| `--no-pass2` | 이미지 pass 끄기 (베이스라인) |
| `--pass2-blind` | 1차 결과를 감추고 독립 판단 (앵커링 편향 비교) |
| `--include-ocr` | 결과 JSON 에 OCR 박스 + LLM 원시응답 포함 (디버깅) |
| `--image-max-side` | pass2 이미지 크기. **1500 이상 유지** (작으면 작은 글씨를 못 읽음) |
| `--cpu` | OCR 을 CPU 로 |

라이브러리로:

```python
from pii_pipeline import PiiPipeline, PipelineConfig

pipeline = PiiPipeline(PipelineConfig())     # 배치 처리 시 인스턴스는 하나만
result = pipeline.run("document.png")
for r in result.regions:
    print(r.type, r.bbox, r.source.value, r.confidence)
```

---

## 5. 출력 형식

```json
{
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

라벨 17종은 `src/pii_pipeline/schema.py` 의 `PII_LABELS` 에서 수정한다.
프롬프트와 스키마가 이 상수에서 자동 생성되므로 한 곳만 고치면 된다.

---

## 6. 아직 검증 안 된 부분

GPU 없는 환경에서 만들었으므로 다음 두 개는 **첫 실행 시 깨질 가능성이 있다.**

| 항목 | 확인할 것 |
|---|---|
| PaddleOCR 연동 | `paddle_runner.py` 의 `parse()` 가 실제 반환 형태와 맞는지. **2.x 기준으로 작성됨** — 3.x 설치하면 손봐야 한다 |
| vLLM 연동 | `guided_json` / `chat_template_kwargs` 가 실제로 먹는지 |

로직·배선은 테스트로 검증됐다.

---

## 참고

- 저장소의 모든 번호는 체크섬으로 합성한 **가짜 값**이다.
- **실제 내부 자료를 커밋하지 말 것.** 검증은 합성 데이터로 한다.
  (`.gitignore` 에 `data/`, `samples/`, `*.pdf` 를 넣어뒀다.)
- 과검 제외 규칙은 `src/pii_pipeline/exclusions.py` 훅에 추가한다.
  현재는 recall 우선으로 아무것도 제외하지 않는다.
