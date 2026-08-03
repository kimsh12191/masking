"""VLM 프롬프트 테스트.

프롬프트는 이 파이프라인의 **탐지기 그 자체**다. 옛 규칙 레이어가 하던 형식
판별이 여기로 옮겨왔으므로, 그 지식이 실제로 문장에 들어 있는지 지킨다.
"""

from __future__ import annotations

from pii_pipeline.llm.prompts import (
    SYSTEM_LOCATE,
    SYSTEM_VLM,
    USER_VLM,
    _REPEATS,
    build_user,
)
from pii_pipeline.schema import PII_LABELS


class TestFixedSystemPrompt:
    def test_is_a_constant_with_no_unfilled_placeholders(self) -> None:
        """가변값이 끼면 vLLM prefix caching 이 타일마다 깨진다.

        중괄호 자체는 있어야 한다 (출력 형식 예시가 JSON 이다). 남아 있으면
        안 되는 것은 f-string 이 채우지 못한 흔적(``{{``)이다.
        """
        assert isinstance(SYSTEM_VLM, str)
        assert "{{" not in SYSTEM_VLM
        assert "}}" not in SYSTEM_VLM

    def test_does_not_depend_on_the_page_or_tile(self) -> None:
        """타일 번호·페이지 크기가 들어가면 호출마다 프롬프트가 달라진다.

        좌표 환산은 ``detect.py`` 가 코드로 한다. 모델에게 "너는 3번 타일이고
        페이지의 60~100% 구간이다" 를 알려주면 환산을 틀리고, 캐싱도 깨진다.
        """
        for banned in ("타일", "tile", "페이지 크기"):
            assert banned not in SYSTEM_VLM, banned

    def test_declares_json_only_output(self) -> None:
        assert "JSON 객체 하나만" in SYSTEM_VLM
        assert '{"findings":[]}' in SYSTEM_VLM.replace(" ", "")

    def test_does_not_hand_over_an_ocr_box_list(self) -> None:
        """회귀 방지: 박스 번호 고르기로 되돌아가면 이 테스트가 깨진다.

        옛 구조의 실패 원인이 정확히 이것이었다 — 시각적 위치를 박스 번호로
        표현하게 만들면 모델이 조용히 틀린 번호를 고르고, 검증할 방법이 없다.
        """
        for banned in ("박스 번호", "[번호]", "이미 탐지된", "놓친 것만", "idx"):
            assert banned not in SYSTEM_VLM, banned


class TestFormatTable:
    """옛 정규식 레이어의 자리수 지식."""

    def test_states_rrn_and_license_digit_counts(self) -> None:
        assert "13자리" in SYSTEM_VLM
        assert "12자리" in SYSTEM_VLM

    def test_forbids_calling_a_13digit_value_a_license(self) -> None:
        """이전 구조 최악의 오류가 이것이었다 (주민등록번호 -> 운전면허번호)."""
        assert "13자리 6-7 형태는 절대 DRIVER_LICENSE 가 아니다" in SYSTEM_VLM

    def test_covers_every_checksummed_identifier(self) -> None:
        for label in ("RRN", "FOREIGN_ID", "CORP_NO", "BIZ_NO", "CARD_NO", "DRIVER_LICENSE"):
            assert label in SYSTEM_VLM, label

    def test_tells_model_to_fall_back_to_other_when_unsure(self) -> None:
        """종류를 못 골라 항목을 버리는 것이 가장 나쁘다."""
        assert "OTHER" in SYSTEM_VLM
        assert "버리는 것이 가장 나쁘다" in SYSTEM_VLM


class TestTranscriptionPolicy:
    def test_asks_for_verbatim_text(self) -> None:
        """감사에서 '문서에 뭐라고 적혀 있었나' 를 되짚어야 한다."""
        assert "적힌 그대로" in SYSTEM_VLM
        assert "원래 값으로 고쳐 쓰지 마라" in SYSTEM_VLM

    def test_type_follows_the_original_value_not_the_evasion(self) -> None:
        assert "원래 값 기준" in SYSTEM_VLM

    def test_lists_evasion_patterns(self) -> None:
        for pattern in ("구1공8공4", "9O1231-1234567", "홍 길 동", "(at)"):
            assert pattern in SYSTEM_VLM, pattern

    def test_warns_about_stopping_early_in_tables(self) -> None:
        """가장 흔한 실패는 표 위쪽 몇 줄만 적고 끝내는 것이다."""
        assert "모든 행을" in SYSTEM_VLM

    def test_excludes_label_only_cells(self) -> None:
        assert "항목명(라벨)만 있는 칸은 보고하지 마라" in SYSTEM_VLM


class TestRepeatedValues:
    """같은 값이 행마다 반복되는 표에서 한 번만 보고하는 실패를 막는다.

    실측 사례: 주민등록표 초본에서 같은 주소가 5행에 반복되는데 2행만
    보고됐다. 값이 같으면 한 번만 말하는 것이 언어모델의 기본 습관이고,
    guided decoding 은 배열 길이만 강제하므로 이것을 막지 못한다.
    """

    def test_demands_every_occurrence(self) -> None:
        assert "같은 값이 여러 곳에 있으면 그 자리를 전부 적어라" in SYSTEM_VLM

    def test_counts_table_rows_separately(self) -> None:
        """"모든 행을 적어라" 만으로는 부족했다.

        그 문장은 "표 아래쪽까지 내려가라" 로 읽히고, 이미 적은 값과 **같은**
        값을 다시 적어야 한다는 뜻으로는 읽히지 않는다. 막으려는 것은 조기
        종료가 아니라 중복 제거다.
        """
        assert "행마다 따로 센다" in SYSTEM_VLM
        assert "합칠 이유가 되지 않는다" in SYSTEM_VLM

    def test_still_allows_a_single_occurrence(self) -> None:
        """과탐 방향으로 밀어붙이지 않는다 — 한 곳에만 있으면 하나만 적는다."""
        assert "한 곳에만 있으면 하나만 적는다" in SYSTEM_VLM

    def test_training_and_inference_share_one_block(self) -> None:
        """복사하면 갈라지고, 갈라진 것은 성능 저하로만 나타나 눈에 띄지 않는다.

        이 문단이 실제로 그렇게 갈라져 있었다 — ``SYSTEM_LOCATE`` 에만 있어서
        가르친 행동을 추론에서 요구하지 않고 있었다.
        """
        assert _REPEATS in SYSTEM_VLM
        assert _REPEATS in SYSTEM_LOCATE


class TestNoticeTextIsNotDroppedWholesale:
    """안내 문구 제외가 그 안의 사람까지 데려가지 않게 한다.

    실측 사례: "이 초본은 ... 틀림없음을 증명합니다" 문단 안의 담당자
    이름·연락처, 신청인 이름·생년월일, 발급기관이 **전부** 누락됐다.
    모델이 제외 규칙("약관·안내 문구")을 블록 단위로 적용한 결과다.
    """

    def test_forbids_dropping_the_whole_block(self) -> None:
        assert "안내 문구를 블록째로 넘기지 마라" in SYSTEM_VLM
        assert "문장 속에 있다고 개인정보가 아닌 것이 되지 않는다" in SYSTEM_VLM

    def test_keeps_the_notice_text_itself_excluded(self) -> None:
        """좁히는 것이지 뒤집는 것이 아니다 — 문구 자체는 여전히 버린다."""
        assert "약관·안내 문구" in SYSTEM_VLM
        assert "문구는 버리고" in SYSTEM_VLM

    def test_shows_worked_examples_for_the_missed_roles(self) -> None:
        """규칙만으로는 모델이 매번 다르게 판단했다."""
        for role in ("담당", "신청인"):
            assert role in SYSTEM_VLM, role

    def test_issuing_office_is_an_org(self) -> None:
        assert "구청장" in SYSTEM_VLM
        assert "발급·확인 기관" in SYSTEM_VLM

    def test_examples_use_placeholder_identities(self) -> None:
        """이 파일은 커밋되는 소스다 — 실제 문서에서 읽은 값을 넣지 않는다.

        가르치려는 것은 서식의 모양뿐이고 자리표시자로 충분하다. 예시에
        진짜 이름·연락처가 들어가면 프롬프트가 개인정보 보관처가 된다.
        """
        assert "홍길동" in SYSTEM_VLM
        assert "031-000-0000" in SYSTEM_VLM


class TestBboxGuidance:
    def test_explicitly_lowers_coordinate_pressure(self) -> None:
        """좌표 정확도를 압박하면 그 용량이 전사 정확도에서 빠져나간다."""
        assert "한 픽셀 단위로 맞추려고 애쓰지 마라" in SYSTEM_VLM
        assert "값을 정확히 읽는 것이 좌표보다 훨씬 중요하다" in SYSTEM_VLM

    def test_uses_the_native_qwen_grounding_format(self) -> None:
        """형식을 우리 편의대로 정하면 grounding 정확도로 대가를 치른다.

        Qwen3-VL 의 기본 좌표계는 0~1000 이고 키는 ``bbox_2d`` 다 (공식 쿡북).
        한때 ``bbox_norm`` + 0.0~1.0 소수를 요구했는데 그건 어느 버전의 native
        형식도 아니었다.
        """
        assert "bbox_2d" in SYSTEM_VLM
        assert "0 이상 1000 이하의 정수" in SYSTEM_VLM
        assert "bbox_norm" not in SYSTEM_VLM
        # 소수를 쓰지 말라고 명시해야 한다 — 반대 방향의 지시가 남아 있으면 안 된다
        assert "소수를 쓰지 마라" in SYSTEM_VLM

    def test_prefers_generous_boxes(self) -> None:
        assert "넉넉하게" in SYSTEM_VLM

    def test_says_ocr_decides_the_final_coordinate(self) -> None:
        assert "최종 좌표는 OCR 이" in SYSTEM_VLM


class TestConfidenceGuide:
    def test_gives_anchors_so_the_model_does_not_answer_0_9_always(self) -> None:
        for anchor in ("0.9 이상", "0.5 ~ 0.8", "0.5 미만"):
            assert anchor in SYSTEM_VLM, anchor


class TestLabelVocabulary:
    def test_every_label_appears(self) -> None:
        for label in PII_LABELS:
            assert label in SYSTEM_VLM, label


class TestUserMessage:
    def test_default_is_constant(self) -> None:
        assert build_user() == USER_VLM

    def test_empty_hint_is_the_default(self) -> None:
        assert build_user("") == USER_VLM
        assert build_user("   ") == USER_VLM

    def test_hint_is_appended(self) -> None:
        out = build_user("가족관계증명서다")
        assert out.startswith(USER_VLM)
        assert "가족관계증명서다" in out

    def test_mentions_tables(self) -> None:
        assert "표" in USER_VLM
