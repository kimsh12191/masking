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

구성: `Qwen3.5-9B` (VLM) + `PaddleOCR` (측량)

상세 설계와 근거는 [`docs/DESIGN.md`](docs/DESIGN.md) 참조.

---

## 0. 파이프라인 — 두 단계다

```
원본 이미지
  ①  전처리       기울기 보정 + 긴 변 정규화          preprocess.py
  ②  VLM 탐지     이미지를 보고 값을 전사             detect.py    → 값 정확, 좌표 대략
  ③  크롭 OCR     지목된 곳을 잘라 배치 인식          locate.py    → 좌표 정확
  ④  검증         체크섬 · 자리수 · 항목명 필터       verify.py
      출력        JSON + 오버레이 PNG                output.py
```

**판단(의미)과 좌표(기하)를 다른 엔진에 맡긴다.** 개인정보 판단은 의미 문제라서
VLM 이 하고, 좌표는 기하 문제라서 OCR 이 한다. VLM 에게 OCR 박스 번호를 고르라고
하지 않고, OCR 에게 무엇이 개인정보인지 묻지 않는다.

이 순서가 중요한 이유는 [`docs/DESIGN.md`](docs/DESIGN.md) 에 적었다. 요약하면:
이전 구조는 OCR(기하)을 먼저 돌리고 그 결과를 의미 판단의 **입력 형식으로
강제**했다. 그래서 VLM 이 "시각적 위치 → 박스 번호" 라는, 자기가 가장 못하는
일을 해야 했고, 틀려도 검증할 방법이 없었다.

### ② VLM 은 전사만 한다

VLM 에게 주는 것은 **이미지 하나뿐**이다. OCR 박스 목록도, "이미 탐지된 번호"도,
좌표 정밀도 요구도 없다.

```json
{"text":"901112-2846261","type":"RRN","field":"가족사항 자녀 주민등록번호",
 "bbox_2d":[420,310,680,340],"conf":0.9}
```

- `text` — 문서에 **적힌 그대로**. 회피 표기면 회피 표기가 담긴다 (감사 추적).
  단 `type` 은 **원래 값 기준**으로 정한다 (`"구1공8공4"` → `PHONE`).
- `bbox_2d` — **크롭용 대략 좌표.** 키 이름과 스케일 모두 **Qwen-VL 의 native
  grounding 형식**이다 — `bbox_2d`, 0~1000 정수 ([공식 쿡북](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)).
  한때 `bbox_norm` 이라는 자체 키에 0.0~1.0 소수를 요구했는데 그건 어느 Qwen
  버전의 native 형식도 아니었다. grounding 은 모델이 분포에 가장 민감한
  과제이고, 9B 급에서 그 대가는 곧바로 좌표 정확도로 나온다.
  그러면서도 프롬프트는 "한 픽셀 단위로 맞추려 애쓰지 마라, 값을 정확히 읽는
  것이 훨씬 중요하다, 넉넉하게 잡아라" 를 명시한다 — 좌표 정확도를 압박하면
  그 용량이 전사 정확도에서 빠져나간다.
- `field` — 그 값이 놓인 칸의 이름. 사람 검수용이고, 같은 값이 여러 곳에
  나올 때 구분에도 쓴다.

**타일링이 기본값(3조각)인 이유.** 한 번의 호출로 페이지 전체를 전사하게 하면
뒤쪽 항목이 잘린다 — 모델이 분류를 못해서가 아니라 긴 구조화 출력의 후반부가
무너지기 때문이다. 가족관계증명서 한 장이 구성원 9명 × 4필드 = 36건인데 9B
모델이 한 번에 온전히 뱉기를 기대할 수 없다. 조각당 10건 내외로 나눈다.

부수 효과가 크다. 2480px 페이지를 통째로 보내면 `image_max_side` 에서 축소돼
주민등록번호 숫자가 10px 대로 떨어진다. 3조각이면 조각 긴 변이 1748 이라
**축소가 아예 일어나지 않는다.** `--print-config` 가 이 조합을 검산해준다.

경계에 걸친 줄을 잃지 않도록 조각을 8% 겹쳐 자른다. 겹침에서 생긴 중복은 코드가
정리하는데, **값이 같아도 위치가 겹치지 않으면 남긴다** — 같은 이름이 문서 여러
곳에 나오는 것은 정상이고 각각 마스킹해야 한다.

### ③ 크롭 OCR 이 좌표를 확정한다

**VLM 의 좌표는 신뢰하지 않는다. VLM 의 판단(위치와 종류)만 쓴다.**

크롭을 넉넉히(35%, 최소 12px) 떠서 2배 업샘플 → 배치 OCR → 크롭 안에서 대상
박스를 고른다. 고른 **OCR 박스의 좌표**가 최종 좌표다.

**이 단계에 모델 호출은 없다.** 선택은 전부 코드가 결정론적으로 한다. 크롭 OCR
결과를 다시 VLM 에게 보여주고 고르라고 하면 이전 구조의 실패로 되돌아가고,
비용도 `O(타일)` 에서 `O(항목)` 으로 등급이 바뀐다 — 항목 25건이면 호출 25회,
페이지가 빽빽할수록(개인정보가 많을수록) 더 비싸지는 최악의 방향이다.

선택 순서 — 좁은 답부터, 확실한 근거부터:

| 순위 | 근거 | 좌표 | `agreement` | `char_span` |
|---|---|---|---|---|
| ① | 박스 1개 안에서 텍스트 완전일치 | 정확 | `exact` | 있음 |
| ② | 인접 박스 이어붙여 완전일치 (`"010-1234"` + `"5678"`) | 정확 | `exact` | 없음 |
| ③ | **VLM bbox 와 위치가 겹치는 박스** (텍스트 안 봄) | 정확 | `none` | 없음 |
| ④ | 크롭에 OCR 박스가 없음 | 근사 | `none` | 없음 |

**③이 이 설계의 핵심이다.** VLM 이 잘하는 일은 "여기에 이 종류의 개인정보가
있다" 는 판단이다 — 근거가 중복적이기 때문이다 (필드 라벨, 표 구조상의 위치,
옆 칸과의 관계). 반면 `901112-2846261` 을 13자리 모두 정확히 맞히는 것은
중복성이 0인 과제이고, 9B 급 모델에서 신뢰할 수 없다. 숫자에는 오독을
교정해줄 언어모델 prior 가 없다 (`홍길둥` 은 `홍길동` 으로 교정되지만
숫자는 안 된다).

①②만 있으면 **잘하는 일의 결과를 못하는 일이 가로막는다.** VLM 이 위치를
정확히 지목했는데 숫자 한 자리를 틀리면 그 판단이 좌표로 이어지지 못한다.
그래서 텍스트 일치는 필수 조건이 아니라 **확인 증거**로 강등하고, 좌표 확정은
기하가 맡는다.

③은 `FAILED` 박스도 후보에 넣는다. det 는 성공하고 rec 만 실패한 박스는
"여기 글자는 있는데 못 읽었다" 는 뜻이고, VLM 이 그 자리를 지목했다면 **그
좌표가 VLM 근사치보다 정확하다.** 손글씨 칸이 여기서 `vlm_coarse` 를
벗어난다 — 텍스트 매칭으로는 불가능했던 회수다.

**③의 대가:** VLM bbox 가 어긋나면 엉뚱한 셀이 조용히 선택된다. 완충 장치는
둘이다 — 가장 좁은 후보를 우선하고, `agreement=none` + `needs_review` +
`reason` 에 양쪽 판독을 함께 남긴다. `--no-geometry-fallback` 으로 끄고 A/B
비교할 수 있다.

좁은 답을 먼저 찾는 것도 중요하다. 이어붙이기나 기하를 먼저 시도하면 값
하나에 옆 칸이 딸려 들어와 박스가 셀 두 개를 덮는다.

**유사 매칭 단계는 없다.** 한때 ②와 ③ 사이에 "글자가 비슷하면 같은 값으로
본다"(임계값 0.65) 는 단계가 있었지만 제거했다. ③이 같은 일을 더 안전하게
한다 — 텍스트가 비슷하면서 위치도 겹치면 ③이 같은 박스를 고르고, 위치가 안
겹치면 애초에 다른 값이라 유사도로 이어붙여선 안 된다. 특히 숫자는 한 글자
다르면 "오독된 같은 번호" 가 아니라 **그냥 다른 사람의 번호**다. 임계값 하나를
없애는 대신 잃는 것이 없다.

①②에서 못 찾으면 크롭을 넓혀 **한 번만** 재시도한다. 그래서 OCR 배치는 두 번
돈다 (전부 → 실패분만). 두 번 넓혀서 안 잡히면 크롭 크기 문제가 아니라
판독 불일치 문제이고, 그건 ③이 받는다.

### 교차검증이 공짜로 나온다

두 엔진이 독립적으로 같은 값을 읽었다면 그 값은 거의 확실하다. 어긋난 경우는
네 가지로 나뉘고, **처방이 각각 다르다.**

| 상황 | `source` | `needs_review` | `reason` |
|---|---|---|---|
| 텍스트가 일치함 | `ocr_refined` | 불일치일 때만 | 일치 방식 |
| 텍스트는 다른데 위치가 겹침 | `ocr_refined` | `true` | **양쪽 판독 기록** |
| 겹치는 박스가 없어 가까운 줄을 골랐음 | `ocr_refined` | `true` | VLM **좌표 밀림** |
| 크롭에 박스가 하나도 없음 | `vlm_coarse` | `true` | OCR **검출** 문제 |
| 읽을 글자가 애초에 없음 (서명) | `vlm_coarse` | `false` | 서명·인영 |

세 번째가 이 파이프라인이 실제로 고장 났던 자리다. 크롭은 VLM 좌표가 어긋난다는
전제로 넉넉히 뜨는데, 박스 선택은 **패딩 없는** VLM bbox 와의 겹침만 봤다. 밀림이
bbox 크기를 넘으면 겹침이 0 이 되어 페이지의 **모든** 항목이 `vlm_coarse` 로
떨어졌다 — 좌표를 교정하려고 만든 단계가 좌표가 부정확할 때 정확히 작동을
멈춘 것이다. 지금은 크롭 범위 안에서 가장 가까운 줄을 고르고, 어느 쪽이 맞는지
모르므로 **VLM 좌표까지 함께 덮는다** (덧칠은 미탐보다 싸다).

두 번째가 진단의 핵심이다.

```
"VLM bbox 와 위치가 겹치는 OCR 박스를 선택 (텍스트는 불일치).
 VLM '901112-2846261' vs OCR '9O1112-284626'"
```

이 한 줄이 로그에 남으면 어느 엔진을 손봐야 하는지 바로 보인다. 이전 구조에서는
이 정보가 그냥 사라졌다.

3번과 4번을 구분하는 것도 중요하다. 처방이 다르다 — 3번은 OCR det 임계값이나
업샘플을 손볼 일이고, 4번은 크롭을 넓혀도 안 낫는다 (더 많은 무관한 박스가
들어올 뿐이다). 프롬프트나 모델 쪽 문제다.

**좌표를 못 잡아도 항목을 버리지 않는다.** `findings` 와 `regions` 는 항상 1:1
이다. 개인정보를 조용히 떨어뜨리는 것이 이 파이프라인에서 가장 위험한 실패다.

### ④ 규칙은 탐지기가 아니라 검증기다

옛 정규식 탐지 레이어(`detectors.py`)를 제거했다. 그 지식은 두 곳으로 갔다.

**자리수·구성 형식은 프롬프트의 형식표로.** 모델은 표를 보면 자리수를 센다.

```
RRN             13자리  6-7        901112-2846261   앞 6자리가 생년월일(YYMMDD)
DRIVER_LICENSE  12자리  2-2-6-2    11-12-345678-01
BIZ_NO          10자리  3-2-5      123-45-67890
...
13자리 6-7 형태는 절대 DRIVER_LICENSE 가 아니다.
```

**체크섬은 사후 검증으로.** VLM 이 뱉은 값에 `validate_rrn` 등을 돌린다.
**두 엔진의 값을 모두 시도**한다 — OCR 이 한 자리 흘렸어도 VLM 이 온전하면
값은 실재한다.

| 검사 | 결과 |
|---|---|
| 체크섬 통과 + 두 엔진 일치 | `conf 1.0`, `needs_review=false` — 사람이 볼 이유가 없다 |
| 체크섬 미통과 | `checksum="failed"`, 검토 대상 |
| 체크섬 없는 라벨 | `checksum=null` — "검증 실패" 와 구분된다 |
| 자리수 불일치 | 검토 대상 + `reason` 에 어느 라벨에 맞는 자리수인지 |
| 13자리 + 주민번호 체크섬 통과인데 다른 라벨 | **type 교정** + 경고 |

마지막 줄이 이전 구조에서 실제로 관측된 오류다. 정규식이 탐지기였을 때는 12자리
숫자열을 "체크섬 통과한 운전면허번호, confidence 1.00, 검토 불필요" 로 확정했고
(`DRIVER_LICENSE` 는 체크섬이 아예 없는데도), 하이픈을 흘린 주민등록번호가 거기로
빨려 들어가 확정 라벨을 받고 이후 판단에서 제외됐다.

**검증기는 틀린 답을 만들 수 없다.** 최악의 경우 "모르겠다" 를 낼 뿐이다.
탐지기였을 때는 틀린 답을 확신을 담아 만들어냈다.

여기에 `is_field_label` 로 `"성 명"` 같은 인쇄된 헤더 셀 오탐도 걸러낸다
(이전에는 이 함수가 LLM 출력에 아예 적용되지 않았다).

---

## 1. 지금 바로 테스트 (GPU/모델 없이)

로직 검증은 의존성 없이 돌아간다.

```bash
git clone <repo> && cd masking
git checkout claude/pass2-vlm-performance-x34cw1

pip install pytest 'numpy<2.0' 'Pillow<11' opencv-python-headless PyYAML pypdfium2
python -m pytest
```

→ **730건 통과**면 체크섬·회피 표기 정규화·타일 좌표 환산·크롭 매칭·검증·
프롬프트·설정 로딩·라벨 스키마·PDF 페이지 분해·파이프라인 배선이 모두 정상이다.
(`numpy`/`Pillow` 미설치 시 skip 이 늘어난다.)

설정만 확인해보려면:

```bash
python scripts/run.py --print-config
```

타일 수와 `image_max_side` 의 조합을 검산해 알려준다.

```
② VLM 탐지  타일 3개 (겹침 8%)  호출 3회  image_max_side=2000  타일 세로 약 826px — 축소 없음
③ 좌표 확정  크롭 패딩 35% (최소 12px)  업샘플 x2.0  기하 fallback=ON
④ 검증      체크섬 교정=ON
```

합성 서식 이미지도 이 상태에서 만들 수 있다 (정답 JSON 포함).

```bash
python scripts/make_synthetic.py --out data/synth --count 3
```

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

`--enable-prefix-caching` 이 이 파이프라인에서는 특히 중요하다. 페이지를 타일로
쪼개 호출하는데 **시스템 프롬프트와 user 메시지가 완전히 고정**이라 (타일 번호도
안 들어간다 — 좌표 환산은 코드가 한다) 이미지 앞까지 캐시가 걸린다. 타일 수를
늘려도 프롬프트 비용은 선형으로 늘지 않는다.

### GPU 가 여러 장이면 — OCR 과 vLLM 을 나눈다

기본값은 둘 다 0 번 GPU 를 쓴다. 카드가 여러 장이면 나누는 편이 낫다.

```bash
# vLLM 은 0 번
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-9B --max-model-len 8192 ...

# OCR 은 1 번 — 셋 중 아무 방법이나
python scripts/run.py sample.png --gpu-id 1     # ① CLI (일회성)
export PII_OCR_GPU_ID=1                          # ② 환경변수
#   config.yaml:  ocr: { gpu_id: 1 }             # ③ 설정 파일
```

번호는 `nvidia-smi` 의 인덱스다. 단, `CUDA_VISIBLE_DEVICES` 를 같이 쓰면
**그 목록 안에서의 상대 번호**로 해석된다 — `CUDA_VISIBLE_DEVICES=2,3` 이면
`--gpu-id 1` 은 물리 2번이 아니라 3번이다. 헷갈리면 둘 중 하나만 쓸 것.

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

`boxes.png` 를 열어 **좌표가 맞는지 눈으로 확인**하는 게 가장 빠르다.
라벨은 박스 **바깥**에 붙으므로 안의 내용을 가리지 않는다.

| 표시 | 의미 |
|---|---|
| 초록 | `ocr_refined` — 크롭 OCR 이 좌표를 확정. 픽셀 단위로 정확 |
| 빨강 | `vlm_coarse` — 크롭 OCR 이 값을 못 찾아 VLM 좌표 사용. 검토 필수 |
| `OK` | 체크섬 통과 |
| `X` | 체크섬 미통과 — OCR 오독 또는 형식 오류 의심 |
| `!` | `needs_review` 플래그 |

**색이 두 개뿐인 것은 의도한 것이다.** 검수자가 눈으로 내려야 하는 판단은
"이 박스 좌표를 믿어도 되나" 하나이고, 그건 이 두 갈래로 끝난다. 판단 근거는
라벨 접미사로 붙인다.

### 마스킹 범위 방침

**넓게 잡는다.** 다음도 모두 탐지 대상이다.

- 문서를 발행·접수한 **기관 자체의 정보** — 은행명, 지점명, 부서명, 기관
  대표번호, 기관 주소, 기관 사업자등록번호
- **문서에 적힌 모든 날짜** — 생년월일은 `BIRTH`, 신청일·계약일·발급일·만기일
  등은 `OTHER`
- 사람·조직에 붙는 **모든 식별번호** — 사번, 고객번호, 관리번호, 증서번호
- 서식에 **인쇄된 예시 값** (`예) 홍길동`)

제외 대상은 **아무도 식별하지 않는 값**뿐이다: 금액, 이자율, 기간, 상품명,
약관·안내 문구. 그리고 값 없이 항목명만 있는 칸.

이 방침은 `llm/prompts.py` 의 `_INCLUSIONS` 에 있다. 좁히려면 거기를 고친다.

---

## 3. 성능을 어떻게 읽는가

단계가 두 개뿐이므로 **각 지표가 어느 단계의 성능인지 1:1 로 대응한다.**
이게 이 구조의 실질적인 이득이다 — 이전 5단계 구조에서는 오차를 어느 단계에
귀속시킬 수 없었다.

```
=== data/synth/synth_001.png ===
  ② VLM 탐지    24건
  ③ 좌표 확정   88%  (영역 24건, 크롭 OCR 박스 61개)
  ④ 불일치      3건  체크섬실패 1건  검토필요 4건
  좌표출처      {"ocr_refined": 21, "vlm_coarse": 3}
  유형별        {"NAME": 9, "RRN": 8, "BIRTH": 5, "ORG": 2}
  소요시간      {"preprocess": 0.3, "detect": 4.1, "locate": 1.8, ...}
```

| 지표 | 뜻 | 나쁘면 손볼 곳 |
|---|---|---|
| `n_findings` | VLM 이 몇 건 찾았나 (recall 의 분자) | `detect.tiles` 를 늘린다 / 프롬프트 |
| `localized_rate` | 그 중 몇 %가 좌표를 확정했나 | `locate.pad_ratio`·`upscale` / 스캔 품질 |
| `n_disagreement` | 두 엔진이 다르게 읽은 건수 | `reason` 을 보고 어느 엔진인지 판단 |

**측정 순서 권장.**

```bash
# ① 정답이 있는 합성 데이터로 recall 을 재고
python scripts/make_synthetic.py --out data/synth --count 20
python scripts/run.py data/synth/*.png -o out/ --include-ocr

# ② 실제 스캔에서 localized_rate 를 본다
python scripts/run.py data/scan/*.png -o out_scan/ --include-ocr

# ③ 설정을 바꿔가며 비교한다 — 사람 라벨이 필요 없다
python scripts/measure.py data/scan/*.png --per-item

# ④ 박스가 밀렸을 때 원인을 가른다 — 문서 내용을 출력하지 않는다
python scripts/diagnose.py data/scan/*.png
```

### 좌표 밀림 진단 — `scripts/diagnose.py`

박스가 눈에 보이게 밀렸을 때 원인은 셋 중 하나고 **처방이 완전히 다르다.**
눈으로는 구분할 수 없으므로 숫자로 가른다.

| 판정 | 신호 | 처방 |
|---|---|---|
| 좌표 규약 오류 | `coord_convention != unit` | 프롬프트를 모델 native 형식에 맞춘다 |
| 배율 어긋남 | 기울기 `k` 가 1 에서 벗어남 | 규약/리사이즈 환산을 본다 |
| 체계적 밀림 | 중앙 오프셋이 한쪽으로 쏠림 | 타일 환산·전처리 회전을 본다 |
| 무작위 grounding 오차 | 쏠림 없이 흩어짐 | 코드로는 못 고친다. 공차 확대 + 학습 |
| 크롭 OCR 이 빈손 | `vlm_coarse` 가 과반 | 좌표가 아니라 OCR 문제다 |

배율과 평행이동은 절편을 함께 추정해 분리하고, 기울기 이탈은 **표준오차보다
뚜렷할 때만** 배율 문제로 부른다 — 흔들림을 배율 오류로 오진하면 아무 문제 없는
좌표 코드를 고치러 가게 된다.

**출력에 문서 내용이 없다.** 값·필드명·파일명·OCR 텍스트를 찍지 않고 건수와
기하 통계, 판정 문구만 낸다. 폐쇄망에서 화면을 그대로 읽어 나와도 안전하도록
설계했고, 그 사실을 `tests/test_diagnose.py::TestNoLeak` 이 검증한다.

### 설정 비교 — `scripts/measure.py`

같은 문서를 여러 설정으로 돌려 **숫자로 비교한다.** 정답이 없으므로 모든 설정이
찾은 것의 **합집합**을 기준셋으로 삼고, 각 설정이 그중 몇 %를 찾았는지
(`상대재현율`)와 그 설정만 찾은 건수(`단독발견`)를 본다.

```
설정      탐지  영역  좌표확정  불일치  검토  상대재현율  단독발견      초
base        24    24      88%       3     4        82%         0    11.4
x3          29    28      86%       5     7        97%         4    14.1
think       27    26      92%       2     3        93%         2    58.7
```

**상대재현율은 상한이 아니다.** 모든 설정이 함께 놓친 것은 기준셋에 없다.

| 설정 | 무엇을 알아보나 |
|---|---|
| `base` | 현재 운영 설정. 나머지는 전부 이것과 비교한다 |
| `x3` | `--samples 3` 의 합집합. 학습 없이 미탐을 줄이는 손잡이가 실제로 듣는가 |
| `think` | **가중치에 능력이 있는가.** 여기서 잘 되면 그 격차는 증류로 옮길 수 있다 |

`think` 는 후속학습을 할지 말지를 가르는 관문이다. `think` 가 `base` 보다 크게
앞서면 능력은 이미 가중치에 있고 끌어내지 못하는 것뿐이므로, think 트레이스를
학습 데이터로 삼는 증류가 다음 단계가 된다. 차이가 없으면 회수할 것이 없다는
뜻이고, 외부 감독(더 큰 교사·OCR 라벨러)이 필요하다. 스크립트가 이 판정을
직접 찍어준다.

`think` 설정은 guided decoding 을 **끈다.** 스키마를 강제하면 모델이 첫 토큰부터
JSON 을 뱉어야 해서 사고 토큰이 나올 자리가 없다 — 켠 줄 알고 재면 `base` 와
같은 것을 재게 된다.

`--include-ocr` 를 켜면 JSON 에 `findings`(VLM 원본 판단)와 `ocr_boxes`(크롭
OCR 결과)가 함께 남는다. **VLM 이 무엇을 읽었는지와 OCR 이 무엇을 읽었는지를
나란히 볼 수 있어야** 어느 단계를 손볼지 판단할 수 있다.

`localized_rate` 가 낮으면 순서대로 확인한다.

1. `reason` 이 "아무 글자도 읽지 못했다" 뿐인가 → 스캔 품질/OCR 문제
2. `reason` 에 OCR 원문이 있는데 값이 다른가 → 오독. `upscale` 을 올려본다
3. `--crop-pad 0.6` 으로 넓혀서 회복되나 → VLM grounding 문제

합성 데이터 정답의 `difficulty` 필드로 난이도별 recall 을 분리 측정할 수 있다.

| `difficulty` | 무엇을 검증하나 |
|---|---|
| `handwriting` | OCR rec 실패 → `vlm_coarse` 로 남는지 |
| `stamp_overlap` | 도장에 가린 글자 |
| `signature` | 자필 서명 (검토 플래그가 붙지 **않아야** 한다) |
| `inline_label` | `담당자: 조민석` — 라벨과 값이 한 박스 |
| `unlabeled_value` | 라벨 없는 `(1978-04-15)` — 형태와 문맥으로만 판단 |

---

## 4. 마스킹 회피 표기

필터를 피하려고 일부러 다르게 적은 값도 개인정보다. 두 겹으로 대응한다.

**① 프롬프트.** VLM 에게 회피 표기 목록을 주고, `text` 는 **적힌 그대로** 적되
`type` 은 **원래 값 기준**으로 정하게 한다.

**② 정규화 (`normalize.py`).** 크롭 안에서 VLM 텍스트와 OCR 텍스트를 비교할 때
양쪽을 정규형으로 접는다. 그래서 VLM 이 `"010-1234-5678"`, OCR 이
`"공1공-1234-5678"` 로 읽어도 같은 값으로 매칭된다.

| 회피 방식 | 예 | 접힌 결과 |
|---|---|---|
| 한글 수사 섞기 | `구1공8공4` | `910804` |
| 전부 한글 수사 | `공일공-일이삼사-오육칠팔` | `010-1234-5678` |
| 숫자 닮은 글자 | `0lO-1234-5678`, `9O1231-1234563` | `010-1234-5678` |
| 전각 문자 | `０１０－１２３４` | `010-1234` |
| 묶음 문자 | `①②③` | `123` |
| zero-width 삽입 | `901231<ZWSP>-1234563` | `901231-1234563` |
| 하이픈 변종 | `901231‑1234567` (U+2011) | `901231-1234567` |
| 이메일 구분자 치환 | `hong(at)hana(dot)com` | `hong@hana.com` |

**치환은 문맥 게이트를 통과해야 적용된다.** `구`/`이`/`사`/`영` 은 한국어에서
매우 흔한 음절이다. 무조건 접으면 `강남구` 가 `강남9` 가 되어 주소 매칭이
망가진다. 게이트는 이렇다.

- 구분자를 뺀 실질 길이 4자 이상
- 실질 문자 중 "숫자이거나 숫자로 접힐 수 있는" 비율 90% 이상
- 숫자가 하나도 없는 순수 한글 수사는 **7자 이상**일 때만 접는다
  (`일이삼사`는 접지 않고 `공일공일이삼사오육칠팔`은 접는다)

그래서 `강남구`, `리스크관리부`, `제3구역`, `영업일 기준 3일`, `Bob`,
`지점에서 처리` 는 모두 그대로 남는다.

---

## 5. 설정

네 단계로 겹쳐 적용된다 (뒤가 앞을 덮는다).

```
코드 기본값 → config.yaml → 환경변수(PII_*) → CLI 플래그
```

섹션 이름이 파이프라인 단계 이름과 1:1 이다.

```yaml
detect:            # ② VLM 탐지
  tiles: 3         # 조각 수. 밀집한 표 문서는 3~4. 1 이면 뒤쪽이 잘린다
  overlap: 0.08    # 조각 간 겹침 (경계에 걸린 줄 보호)

locate:            # ③ 좌표 확정
  pad_ratio: 0.35  # 크롭 여유. 좁으면 값이 크롭 밖으로 나간다
  upscale: 2.0     # 크롭 업샘플. 작은 글씨 대응

verify:            # ④ 검증
  retype_on_checksum: true   # 체크섬으로 증명되면 type 교정

llm:
  image_max_side: 2000       # detect.tiles 와 함께 봐야 하는 값
  max_tokens: 2048
```

> **`llm.image_max_side` 와 `detect.tiles` 는 따로 보면 안 된다.** 둘의 조합이
> VLM 이 실제로 보는 글자 크기를 결정한다. A4 를 `target_long_side=2480` 으로
> 전처리하면 1748x2480 인데, `tiles=1` 이면 여기서 0.73배로 줄어 주민등록번호
> 숫자가 10px 대가 되고 **읽을 수 없다.** `tiles=3` 이면 조각 긴 변이 1748 이라
> 축소가 없다. `--print-config` 가 검산해준다.

> **`max_tokens` 를 512 로 내리지 말 것.** 한 항목이 `text`/`field`/`bbox_2d`
> 까지 50~70토큰이다. 잘리면 그 타일의 탐지가 **전량** 날아간다 (JSON 이
> 불완전해짐). 잘림은 상한을 올려야만 벗어난다 — `temperature=0` 이라 같은
> 상한으로 재시도하면 매번 같은 지점에서 똑같이 잘린다.

**오타난 설정 키는 조용히 무시되지 않고 즉시 실패한다.** 폐쇄망에서 설정이
반영되지 않은 채 도는 것이 가장 찾기 어려운 실패다.

```
설정 오류: 설정 섹션 'llm' 에 알 수 없는 키가 있습니다: modle
  사용 가능한 키: api_key, base_url, enable_thinking, ..., model, ...
```

루트에 `config.yaml` 을 두면 자동으로 잡힌다 (`.gitignore` 에 있으므로 커밋되지
않는다). `config/default.yaml` 은 템플릿이다.

---

## 6. 주요 옵션

```bash
python scripts/run.py --help
```

| 옵션 | 용도 |
|---|---|
| `--print-config` | 적용된 설정을 출력하고 종료. 타일/해상도 조합을 검산해준다 |
| `--config` | 설정 파일 경로 |
| `--tiles` | 페이지를 몇 조각으로 나눠 VLM 을 호출할지 (기본 3) |
| `--tile-overlap` | 조각 간 겹침 비율 (기본 0.08) |
| `--hint` | 문서 종류 힌트. **비워 두는 것이 기본** (prefix caching 유지) |
| `--samples` | 타일당 호출 횟수 (기본 1). 합집합으로 미탐↓ 과탐↑. 호출은 `tiles × samples` |
| `--sample-temperature` | `--samples 2` 이상일 때의 온도 (기본 0.3). 0 이면 같은 답만 온다 |
| `--crop-pad` | 크롭 여유 비율 (기본 0.35) |
| `--crop-upscale` | 크롭 업샘플 배율 (기본 2.0). `1.0` 이면 끈다 |
| `--no-geometry-fallback` | 기하 선택을 끈다 (텍스트 불일치 = 좌표 포기). A/B 비교용 |
| `--image-max-side` | VLM 이미지 크기. `--tiles` 와 함께 볼 것 |
| `--include-ocr` | JSON 에 `findings` + `ocr_boxes` + 원시응답 포함 (**진단 필수**) |
| `--show-ocr-boxes` | 크롭 OCR 박스 전체를 회색으로 표시 |
| `--no-image` | 박스 표시 이미지 생략, JSON 만 저장 |
| `--pages` | PDF 페이지 범위. `1-3,7` / `5-` |
| `--pdf-password` | 암호화된 PDF 암호 (`PII_PDF_PASSWORD` 환경변수 권장) |
| `--font` | 한글 폰트 경로 (없어도 라벨은 ASCII 라 표시됨) |
| `--gpu-id` | OCR 을 돌릴 GPU 번호 (기본 0) |
| `--cpu` | OCR 을 CPU 로 |

라이브러리로:

```python
from pii_pipeline import PiiPipeline, load_config, save_result

config = load_config()                       # config.yaml + 환경변수 적용
pipeline = PiiPipeline(config.pipeline)      # 배치 처리 시 인스턴스는 하나만

result = pipeline.run("document.png")
save_result(result, config.output.out_dir)   # boxes.png + json 저장

for r in result.regions:
    print(r.type, r.bbox, r.source.value, r.confidence, r.agreement.value)

# 진단 — VLM 이 읽은 것과 OCR 이 읽은 것을 나란히
for r in result.regions:
    if r.agreement.value != "exact":
        print(f"{r.type}: VLM {r.vlm_text!r} / OCR {r.text!r} — {r.reason}")
```

설정 파일 없이 코드로만 구성해도 된다:

```python
from pii_pipeline import DetectConfig, LocateConfig, PiiPipeline, PipelineConfig
from pii_pipeline.llm.client import LlmConfig

pipeline = PiiPipeline(PipelineConfig(
    llm=LlmConfig(model="/opt/models/qwen35-9b", base_url="http://10.0.0.5:8000/v1"),
    detect=DetectConfig(tiles=4),
    locate=LocateConfig(upscale=2.5),
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

## 7. 출력 형식

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
      "source": "ocr_refined",
      "confidence": 1.0,
      "text": "901231-1234563",
      "vlm_text": "901231-1234563",
      "field": "본인 주민등록번호",
      "member_boxes": [[414, 782, 688, 810]],
      "member_index": [3],
      "char_span": [0, 14],
      "verified": true,
      "checksum": "ok",
      "agreement": "exact",
      "coarse": false,
      "needs_review": false,
      "reason": "크롭 OCR 과 완전일치; 체크섬 통과"
    }
  ],
  "timings": {"preprocess": 0.31, "detect": 4.12, "locate": 1.83,
              "verify": 0.01, "total": 6.27},
  "stats": {"n_findings": 24, "n_regions": 24, "localized_rate": 0.875,
            "n_disagreement": 3, "n_checksum_failed": 1, "n_needs_review": 4,
            "by_source": {"ocr_refined": 21, "vlm_coarse": 3}}
}
```

- `source` — **좌표를 어떻게 얻었는가.** `ocr_refined`(정확) /
  `vlm_coarse`(근사, 검토 필수). 판단은 전부 VLM 이므로 "출처" 가 아니라 좌표
  획득 경로를 구분한다. 다운스트림이 신뢰할지 결정하는 기준이 이것이다.
- `text` / `vlm_text` — 각각 크롭 OCR 이 읽은 값과 VLM 이 읽은 값. **둘 다
  남긴다.** 어긋났을 때 어느 엔진 문제인지 판단할 수 있어야 한다.
  회피 표기로 적혀 있었으면 회피 표기가 담긴다 (감사 추적).
- `agreement` — `exact` / `similar` / `none`. 두 엔진의 교차검증 결과.
- `verified` / `checksum` — 체크섬 통과 여부. `checksum: null` 은 "그 라벨에
  체크섬이 없다" 이고 `"failed"` 와 구분된다.
- `field` — VLM 이 준 칸 이름. 검수 화면에서 값의 위치를 설명하는 데 쓴다.
- `member_boxes` — 주소처럼 여러 박스에 걸친 항목의 구성 박스.
  통짜/조각별 마스킹을 선택할 수 있다.
- `char_span` — `text` 내 값의 문자 오프셋. 부분 마스킹(`901231-1******`)
  구현용. **`bbox` 는 항상 박스 전체**임에 유의. 같은 박스에 두 종류가 섞여
  있으면 영역이 두 개 나온다 (`"홍길동 901231-1234563"` → `NAME` + `RRN`).
  부분 마스킹을 구현할 때 **박스의 모든 영역을 함께 적용**해야 한다. 하나만
  쓰면 나머지가 노출된다.
- `id` — 읽기 순서 기반 결정론적 부여. 재처리 시 같은 ID → 어노테이션 재연결 가능.
- `page_no` — PDF 의 1-기반 페이지 번호. 단일 이미지 입력이면 `null`.

`--include-ocr` 를 주면 `findings`(VLM 원본 판단)와 `ocr_boxes`(크롭 OCR 결과,
페이지 좌표), `raw_llm`(타일별 원시응답)이 추가된다.

---

## 8. 개인정보 라벨 (18종)

탐지는 **전부 VLM** 이 한다. 검증만 라벨에 따라 다르다.

### 핵심 9종

| 라벨 | 항목 | 검증 |
|---|---|---|
| `NAME` | 이름 | — (유사 매칭 허용) |
| `RRN` | 주민등록번호 | **체크섬** + 13자리 |
| `ADDRESS` | 주소 | — (유사 매칭 허용) |
| `EMAIL` | 이메일 | — |
| `IP` | IP 주소 | — |
| `ACCOUNT_NO` | 계좌번호 | 없음 (표준 체크섬이 없다) |
| `CARD_NO` | 카드번호 | **Luhn** + 15~16자리 |
| `PHONE` | 전화번호 | — |
| `PASSPORT` | 여권번호 | — |

### 추가 9종

| 라벨 | 항목 | 검증 |
|---|---|---|
| `FOREIGN_ID` | 외국인등록번호 | **체크섬** + 13자리 (2020-10 이후 발급분은 체크섬 없음) |
| `DRIVER_LICENSE` | 운전면허번호 | 12자리 |
| `BIZ_NO` | 사업자등록번호 | **체크섬** + 10자리 |
| `CORP_NO` | 법인등록번호 | **체크섬** + 13자리 |
| `BIRTH` | 생년월일 | — |
| `ORG` | 소속/직장명 | — (유사 매칭 허용) |
| `TITLE` | 직위/직책 | — (유사 매칭 허용) |
| `SIGNATURE` | 서명/인영 | — (텍스트 없음. 항상 `vlm_coarse`) |
| `OTHER` | 위에 없는 개인식별정보 | — |

`RRN` / `FOREIGN_ID` / `CORP_NO` 는 **모두 13자리 6-7 로 형태가 같다.**
프롬프트 형식표가 구분 근거(생년월일부, 성별코드 5~8, 법인 여부)를 주고,
체크섬이 사후에 검산한다. 셋 사이의 혼동은 셋 다 마스킹 대상이라 경미하므로
`retype` 대상이 아니다. **13자리 ↔ 12자리 혼동만 교정한다** — 그게 실제로
관측된 위험한 오류다.

`SIGNATURE` 는 읽을 글자가 없으므로 좌표 확정이 불가능하다. `vlm_coarse` 로
남지만 `needs_review` 는 붙지 않는다 — 검토 큐가 서명으로 가득 차면 정작 봐야
할 불일치 건이 묻힌다.

### 라벨을 바꾸려면

`src/pii_pipeline/schema.py` 의 `PII_LABELS` **한 곳만** 고치면 된다.
프롬프트와 guided-decoding 스키마가 이 상수에서 자동 생성되고,
`tests/test_schema.py` 가 누락을 잡아준다.

체크섬이 있는 정형 식별자를 추가하면 `rules/checksums.py` 의 `VALIDATORS` 와
`verify.py` 의 `DIGIT_LENGTHS` 에도 넣는다. **`detectors.py` 는 없다** —
정규식 탐지 레이어를 제거했으므로 패턴을 추가할 곳이 없고, 프롬프트의 형식표
(`llm/prompts.py` 의 `_FORMATS`)에 한 줄 적으면 된다.

---

## 9. 아직 검증 안 된 부분

GPU 없는 환경에서 만들었으므로 다음은 **첫 실행 시 깨질 가능성이 있다.**

| 항목 | 확인할 것 |
|---|---|
| PaddleOCR 연동 | `paddle_runner.py` 의 `parse()` 가 실제 반환 형태와 맞는지. **2.x 기준** — 3.x 설치하면 손봐야 한다 |
| 크롭 OCR 실측 | 작은 크롭에서 det 가 박스를 만드는지. 못 만들면 `pad_ratio`/`upscale` 을 올려야 한다 |
| vLLM 연동 | `guided_json` / `chat_template_kwargs` 가 실제로 먹는지 |
| 타일 수 | 3 이 적정한지. 문서 밀도에 따라 다르다 — `n_findings` 로 실측할 것 |

로직·배선은 730건의 테스트로 검증됐다. PDF 렌더링은 실제 pypdfium2 로 검증했다.

### 알려진 개선 여지

**전자 PDF.** 전자적으로 생성된 PDF(스캔이 아닌)는 **텍스트 레이어에 정확한
좌표가 이미 들어 있다.** 그런 문서는 ③ 단계에서 OCR 대신 텍스트 레이어를 쓰면
완벽하고 훨씬 빠르다. 금융 문서는 전자 생성 비중이 높으니 실측해볼 값이 있다.
`src/pii_pipeline/pdf.py` 에 같은 내용을 주석으로 남겨뒀다.

**원본 해상도 크롭.** 현재는 `target_long_side` 로 줄인 이미지에서 크롭한다.
원본 스캔이 2480 보다 크면 `target_long_side: null` 로 두는 편이 크롭 OCR 에
유리하다 (다만 VLM 입력이 커지므로 `detect.tiles` 를 함께 늘려야 한다).

---

## 참고

- 저장소의 모든 번호는 체크섬으로 합성한 **가짜 값**이다.
- **실제 내부 자료를 커밋하지 말 것.** 검증은 합성 데이터로 한다.
  (`.gitignore` 에 `data/`, `samples/`, `*.pdf` 를 넣어뒀다.)
