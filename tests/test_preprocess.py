"""전처리 테스트.

opencv 가 없는 환경에서도 돌아야 하므로, cv2 를 직접 쓰는 경로는 테스트하지
않는다. 여기서 보는 것은 **보조 기능의 실패가 페이지를 삼키지 않는지**다.
"""

from __future__ import annotations

import logging

import pytest

from pii_pipeline import preprocess as pp


class TestSkewFailureIsNotFatal:
    """기울기 추정 실패로 그 입력의 마스킹 결과가 사라지면 안 된다.

    현장 사고: ``_estimate_skew`` 안에서 예외가 나 ``run.py`` 의 배치 루프
    (``except Exception`` -> 다음 파일)가 그 페이지를 통째로 버렸다. 결과
    파일이 아예 안 나오므로 **누락을 눈치채기도 어렵다.**
    """

    def test_returns_zero_instead_of_raising(self, monkeypatch) -> None:
        def boom(_gray: object) -> float:
            raise RuntimeError("minAreaRect 실패")

        monkeypatch.setattr(pp, "_estimate_skew_unsafe", boom)
        assert pp._estimate_skew(object()) == 0.0

    def test_warns_so_it_is_not_silent(self, monkeypatch, caplog) -> None:
        def boom(_gray: object) -> float:
            raise ValueError("깨진 입력")

        monkeypatch.setattr(pp, "_estimate_skew_unsafe", boom)
        with caplog.at_level(logging.WARNING):
            pp._estimate_skew(object())
        assert "기울기 추정 실패" in caplog.text
        assert "깨진 입력" in caplog.text

    @pytest.mark.parametrize("exc", [MemoryError, RuntimeError, ValueError])
    def test_any_failure_type_is_absorbed(self, monkeypatch, exc) -> None:
        monkeypatch.setattr(
            pp, "_estimate_skew_unsafe", lambda _g: (_ for _ in ()).throw(exc("x"))
        )
        assert pp._estimate_skew(object()) == 0.0
