"""① 전처리 테스트 — 좌표계의 기준을 만드는 단계다.

여기서 페이지 크기가 정해지고, 그 위에서 타일이 잘리고, 그 안에서 0~1000
좌표가 해석된다. **이 단계가 흔들리면 그 뒤가 전부 흔들린다.**
"""

from __future__ import annotations

import pytest

from pii_pipeline.preprocess import preprocess_array

np = pytest.importorskip("numpy", reason="numpy 미설치 환경에서는 건너뛴다")
pytest.importorskip("cv2", reason="opencv 미설치 환경에서는 건너뛴다")


class TestFixedCanvas:
    """``canvas`` 를 주면 **모든 페이지가 정확히 같은 크기**가 된다.

    좌표 학습의 전제다. 페이지가 고정이라야 타일도 고정이고, 같은 0~1000 값이
    항상 같은 픽셀을 가리킨다.
    """

    CANVAS = (1760, 2464)

    def run(self, w: int, h: int):
        return preprocess_array(
            np.full((h, w, 3), 255, dtype=np.uint8),
            deskew=False,
            canvas=self.CANVAS,
            align=32,
        )

    def test_every_input_size_lands_on_the_same_canvas(self) -> None:
        for w, h in [(1748, 2480), (2480, 3508), (900, 1200), (3508, 2480), (100, 100)]:
            out = self.run(w, h)
            assert (out.width, out.height) == self.CANVAS, f"{w}x{h} 가 캔버스를 벗어났다"

    def test_small_inputs_are_enlarged(self) -> None:
        """축소만 하면 저해상도 스캔이 구석에 작게 박혀 글자 크기가 제각각이 된다."""
        out = self.run(900, 1200)
        assert out.transform["scale"] > 1.0

    def test_aspect_ratio_is_preserved(self) -> None:
        """늘려서 맞추면 글자가 찌그러진다. 여백으로 채워야 한다."""
        out = self.run(1000, 1000)  # 정사각 -> 세로 캔버스
        # 캔버스가 세로로 길므로 **폭**이 먼저 차고 아래가 남는다.
        pad_w, pad_h = out.transform["canvas_pad"]
        assert pad_w == 0 and pad_h > 0
        assert out.transform["scale"] == pytest.approx(self.CANVAS[0] / 1000)

    def test_mismatched_aspect_is_warned(self) -> None:
        """여백이 많으면 그만큼 글자가 작아진다. 조용히 넘기면 원인을 못 찾는다."""
        out = self.run(3508, 2480)  # 가로 문서
        assert any("경고" in a for a in out.applied)

    def test_align_padding_does_not_run_on_top(self) -> None:
        """캔버스가 이미 고정 크기다. 여기서 또 붙이면 캔버스가 아니게 된다."""
        out = self.run(1748, 2480)
        assert "align_pad" not in out.transform

    def test_transform_records_the_inverse(self) -> None:
        """원본 좌표로 되돌리려면 배율과 여백이 남아 있어야 한다."""
        out = self.run(880, 1232)
        assert out.transform["scale"] == pytest.approx(2.0)
        assert out.transform["canvas"] == [1760, 2464]
        assert out.transform["canvas_pad"] == [0, 0]

    def test_rejects_a_bad_canvas(self) -> None:
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        with pytest.raises(ValueError, match="양수"):
            preprocess_array(img, canvas=(0, 100), deskew=False)
        with pytest.raises(ValueError, match="두 값"):
            preprocess_array(img, canvas=(1760,), deskew=False)
