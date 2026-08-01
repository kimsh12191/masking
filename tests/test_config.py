"""설정 로딩 테스트.

적용 순서: 코드 기본값 → 설정 파일 → 환경변수 → (CLI 는 run.py 담당)

``find_config()`` 가 cwd 를 탐색하므로 대부분의 테스트는 ``monkeypatch.chdir``
로 빈 디렉터리에 격리한다. 그렇지 않으면 저장소의 ``config/default.yaml`` 이
자동으로 잡혀 테스트가 환경에 의존한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pii_pipeline.config import (
    ENV_MAP,
    SECTIONS,
    AppConfig,
    OutputConfig,
    describe,
    find_config,
    load_config,
)

pytest.importorskip("yaml", reason="PyYAML 미설치 환경에서는 건너뛴다")

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """빈 작업 디렉터리 + PII_* 환경변수 제거."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PII_CONFIG", raising=False)
    for name in ENV_MAP:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestDefaults:
    def test_no_config_file_uses_code_defaults(self, isolated: Path) -> None:
        config = load_config()
        assert config.source_path is None
        assert config.pipeline.llm.model == "Qwen/Qwen3.5-9B"
        assert config.pipeline.detect.tiles == 3
        assert config.output.write_image is True

    def test_thinking_is_off_by_default(self, isolated: Path) -> None:
        """추론 모드가 켜지면 지연시간 예산을 날린다."""
        assert load_config().pipeline.llm.enable_thinking is False

    def test_temperature_is_zero_by_default(self, isolated: Path) -> None:
        assert load_config().pipeline.llm.temperature == 0.0


class TestFindConfig:
    def test_returns_none_when_nothing_found(self, isolated: Path) -> None:
        assert find_config() is None

    def test_finds_config_yaml_in_cwd(self, isolated: Path) -> None:
        write_yaml(isolated / "config.yaml", "llm:\n  model: a/b\n")
        assert find_config() == Path("config.yaml")

    def test_prefers_cwd_config_over_config_dir(self, isolated: Path) -> None:
        (isolated / "config").mkdir()
        write_yaml(isolated / "config" / "default.yaml", "llm:\n  model: dir/model\n")
        write_yaml(isolated / "config.yaml", "llm:\n  model: cwd/model\n")
        assert load_config().pipeline.llm.model == "cwd/model"

    def test_falls_back_to_config_dir(self, isolated: Path) -> None:
        (isolated / "config").mkdir()
        write_yaml(isolated / "config" / "default.yaml", "llm:\n  model: dir/model\n")
        assert load_config().pipeline.llm.model == "dir/model"

    def test_explicit_path_wins(self, isolated: Path) -> None:
        write_yaml(isolated / "config.yaml", "llm:\n  model: auto/model\n")
        other = write_yaml(isolated / "other.yaml", "llm:\n  model: explicit/model\n")
        assert load_config(other).pipeline.llm.model == "explicit/model"

    def test_missing_explicit_path_raises(self, isolated: Path) -> None:
        with pytest.raises(FileNotFoundError, match="찾을 수 없습니다"):
            load_config("nope.yaml")

    def test_env_config_path_used(self, isolated: Path, monkeypatch) -> None:
        path = write_yaml(isolated / "env.yaml", "llm:\n  model: env/model\n")
        monkeypatch.setenv("PII_CONFIG", str(path))
        assert load_config().pipeline.llm.model == "env/model"

    def test_missing_env_config_path_raises(self, isolated: Path, monkeypatch) -> None:
        monkeypatch.setenv("PII_CONFIG", "/nonexistent/x.yaml")
        with pytest.raises(FileNotFoundError, match="PII_CONFIG"):
            load_config()


class TestFileApplication:
    def test_applies_all_sections(self, isolated: Path) -> None:
        path = write_yaml(
            isolated / "c.yaml",
            "llm:\n  model: local/qwen\n  image_max_side: 1600\n"
            "ocr:\n  lang: en\n  use_gpu: false\n"
            "detect:\n  tiles: 4\n"
            "locate:\n  upscale: 1.0\n"
            "verify:\n  dedup_iou: 0.9\n"
            "pipeline:\n  target_long_side: 1600\n"
            "output:\n  out_dir: result\n  write_image: false\n",
        )
        c = load_config(path)
        assert c.pipeline.llm.model == "local/qwen"
        assert c.pipeline.llm.image_max_side == 1600
        assert c.pipeline.ocr.lang == "en"
        assert c.pipeline.ocr.use_gpu is False
        assert c.pipeline.detect.tiles == 4
        assert c.pipeline.locate.upscale == 1.0
        assert c.pipeline.verify.dedup_iou == 0.9
        assert c.pipeline.target_long_side == 1600
        assert c.output.out_dir == "result"
        assert c.output.write_image is False

    def test_partial_file_leaves_other_defaults(self, isolated: Path) -> None:
        path = write_yaml(isolated / "c.yaml", "llm:\n  model: only/this\n")
        c = load_config(path)
        assert c.pipeline.llm.model == "only/this"
        assert c.pipeline.llm.base_url == "http://127.0.0.1:8000/v1"  # 기본값 유지
        assert c.pipeline.detect.tiles == 3

    def test_records_source_path(self, isolated: Path) -> None:
        path = write_yaml(isolated / "c.yaml", "llm:\n  model: x/y\n")
        assert load_config(path).source_path == path

    def test_empty_file_is_ok(self, isolated: Path) -> None:
        path = write_yaml(isolated / "c.yaml", "")
        assert load_config(path).pipeline.llm.model == "Qwen/Qwen3.5-9B"

    def test_null_section_is_ok(self, isolated: Path) -> None:
        path = write_yaml(isolated / "c.yaml", "llm:\nocr:\n")
        assert load_config(path).pipeline.ocr.lang == "korean"


class TestValidation:
    """오타난 키가 조용히 무시되면 폐쇄망에서 가장 찾기 어려운 실패가 된다."""

    def test_unknown_key_raises_with_suggestions(self, isolated: Path) -> None:
        path = write_yaml(isolated / "c.yaml", "llm:\n  modle: typo\n")
        with pytest.raises(ValueError) as exc:
            load_config(path)
        assert "modle" in str(exc.value)
        assert "model" in str(exc.value)  # 사용 가능한 키 목록에 포함

    def test_unknown_section_raises(self, isolated: Path) -> None:
        path = write_yaml(isolated / "c.yaml", "llmm:\n  model: x\n")
        with pytest.raises(ValueError, match="알 수 없는 최상위 섹션"):
            load_config(path)

    def test_non_mapping_top_level_raises(self, isolated: Path) -> None:
        path = write_yaml(isolated / "c.yaml", "- a\n- b\n")
        with pytest.raises(ValueError, match="매핑이어야 합니다"):
            load_config(path)

    def test_non_mapping_section_raises(self, isolated: Path) -> None:
        path = write_yaml(isolated / "c.yaml", "llm: notamapping\n")
        with pytest.raises(ValueError, match="매핑이어야 합니다"):
            load_config(path)


class TestEnvOverrides:
    def test_env_overrides_file(self, isolated: Path, monkeypatch) -> None:
        path = write_yaml(isolated / "c.yaml", "llm:\n  model: from/file\n")
        monkeypatch.setenv("PII_LLM_MODEL", "from/env")
        assert load_config(path).pipeline.llm.model == "from/env"

    def test_ocr_dirs_from_env(self, isolated: Path, monkeypatch) -> None:
        monkeypatch.setenv("PII_OCR_DET_DIR", "/opt/det")
        monkeypatch.setenv("PII_OCR_REC_DIR", "/opt/rec")
        monkeypatch.setenv("PII_OCR_CLS_DIR", "/opt/cls")
        ocr = load_config().pipeline.ocr
        assert (ocr.det_model_dir, ocr.rec_model_dir, ocr.cls_model_dir) == (
            "/opt/det", "/opt/rec", "/opt/cls",
        )

    def test_gpu_id_from_env_is_int(self, isolated: Path, monkeypatch) -> None:
        """환경변수는 문자열이지만 PaddleOCR 은 정수를 받아야 한다."""
        monkeypatch.setenv("PII_OCR_GPU_ID", "1")
        gpu_id = load_config().pipeline.ocr.gpu_id
        assert gpu_id == 1
        assert isinstance(gpu_id, int)

    def test_bad_gpu_id_raises(self, isolated: Path, monkeypatch) -> None:
        monkeypatch.setenv("PII_OCR_GPU_ID", "첫번째")
        with pytest.raises(ValueError, match="gpu_id"):
            load_config()

    def test_negative_gpu_id_raises(self, isolated: Path, monkeypatch) -> None:
        monkeypatch.setenv("PII_OCR_GPU_ID", "-1")
        with pytest.raises(ValueError, match="gpu_id"):
            load_config()

    def test_base_url_from_env(self, isolated: Path, monkeypatch) -> None:
        monkeypatch.setenv("PII_LLM_BASE_URL", "http://10.0.0.5:8000/v1")
        assert load_config().pipeline.llm.base_url == "http://10.0.0.5:8000/v1"

    def test_use_env_false_ignores_env(self, isolated: Path, monkeypatch) -> None:
        monkeypatch.setenv("PII_LLM_MODEL", "from/env")
        assert load_config(use_env=False).pipeline.llm.model == "Qwen/Qwen3.5-9B"

    def test_empty_env_var_does_not_override(self, isolated: Path, monkeypatch) -> None:
        monkeypatch.setenv("PII_LLM_MODEL", "")
        assert load_config().pipeline.llm.model == "Qwen/Qwen3.5-9B"


class TestShippedConfig:
    """저장소에 포함된 config/default.yaml 이 실제로 로드되어야 한다."""

    def test_loads_without_error(self) -> None:
        config = load_config(REPO_ROOT / "config" / "default.yaml", use_env=False)
        assert config.pipeline.llm.model == "Qwen/Qwen3.5-9B"
        assert config.pipeline.llm.enable_thinking is False
        assert config.pipeline.detect.tiles == 3

    def test_covers_every_section(self) -> None:
        import yaml

        data = yaml.safe_load(
            (REPO_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
        )
        assert set(data) == set(SECTIONS)


class TestDescribe:
    def test_mentions_model_and_endpoint(self, isolated: Path) -> None:
        text = describe(load_config())
        assert "Qwen/Qwen3.5-9B" in text
        assert "127.0.0.1:8000" in text

    def test_shows_no_config_file(self, isolated: Path) -> None:
        assert "기본값 사용" in describe(load_config())

    def test_shows_tile_count(self, isolated: Path) -> None:
        assert "타일 3개" in describe(load_config())

    def test_checks_the_tile_and_image_side_combination(self, isolated: Path) -> None:
        """둘의 조합이 VLM 이 보는 글자 크기를 결정한다. 한쪽만 바꾸면 헤맨다."""
        config = load_config()
        assert "축소 없음" in describe(config)

        config.pipeline.detect.tiles = 1  # 2480px 을 통째로 -> 2000 으로 축소된다
        assert "축소 발생" in describe(config)

    def test_shows_locate_and_verify_settings(self, isolated: Path) -> None:
        text = describe(load_config())
        assert "업샘플" in text
        assert "체크섬 교정" in text


class TestOutputConfigDefaults:
    def test_image_on_by_default(self) -> None:
        assert OutputConfig().write_image is True

    def test_debug_payload_off_by_default(self) -> None:
        assert OutputConfig().include_ocr is False


# --------------------------------------------------------------------------
# CLI 오버레이 (④ 단계)
# --------------------------------------------------------------------------

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run import apply_cli_overrides, build_parser  # noqa: E402


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


class TestCliOverrides:
    def test_unspecified_flags_leave_config_untouched(self, isolated: Path) -> None:
        path = write_yaml(
            isolated / "c.yaml",
            "llm:\n  model: from/file\ndetect:\n  tiles: 4\n",
        )
        config = load_config(path)
        apply_cli_overrides(config, parse(["x.png"]))
        assert config.pipeline.llm.model == "from/file"
        assert config.pipeline.detect.tiles == 4

    def test_cli_overrides_file(self, isolated: Path) -> None:
        path = write_yaml(isolated / "c.yaml", "llm:\n  model: from/file\n")
        config = load_config(path)
        apply_cli_overrides(config, parse(["x.png", "--model", "from/cli"]))
        assert config.pipeline.llm.model == "from/cli"

    def test_cli_overrides_env(self, isolated: Path, monkeypatch) -> None:
        monkeypatch.setenv("PII_LLM_MODEL", "from/env")
        config = load_config()
        apply_cli_overrides(config, parse(["x.png", "--model", "from/cli"]))
        assert config.pipeline.llm.model == "from/cli"

    def test_tiles_override(self, isolated: Path) -> None:
        config = load_config()
        apply_cli_overrides(config, parse(["x.png", "--tiles", "5"]))
        assert config.pipeline.detect.tiles == 5

    def test_locate_overrides(self, isolated: Path) -> None:
        config = load_config()
        apply_cli_overrides(
            config,
            parse(["x.png", "--crop-pad", "0.5", "--crop-upscale", "1.0",
                   "--no-geometry-fallback"]),
        )
        assert config.pipeline.locate.pad_ratio == 0.5
        assert config.pipeline.locate.upscale == 1.0
        assert config.pipeline.locate.geometry_fallback is False

    def test_upscale_one_still_overrides(self, isolated: Path) -> None:
        """1.0 은 '업샘플 끄기'라는 뜻이고 '미지정'이 아니다."""
        path = write_yaml(isolated / "c.yaml", "locate:\n  upscale: 3.0\n")
        config = load_config(path)
        apply_cli_overrides(config, parse(["x.png", "--crop-upscale", "1.0"]))
        assert config.pipeline.locate.upscale == 1.0

    def test_no_image_turns_off_image_output(self, isolated: Path) -> None:
        config = load_config()
        apply_cli_overrides(config, parse(["x.png", "--no-image"]))
        assert config.output.write_image is False

    def test_cpu_flag_disables_gpu(self, isolated: Path) -> None:
        config = load_config()
        apply_cli_overrides(config, parse(["x.png", "--cpu"]))
        assert config.pipeline.ocr.use_gpu is False

    def test_gpu_id_override(self, isolated: Path) -> None:
        config = load_config()
        apply_cli_overrides(config, parse(["x.png", "--gpu-id", "1"]))
        assert config.pipeline.ocr.gpu_id == 1

    def test_gpu_id_zero_still_overrides(self, isolated: Path) -> None:
        """0 은 falsy 지만 '미지정'이 아니다."""
        path = write_yaml(isolated / "c.yaml", "ocr:\n  gpu_id: 3\n")
        config = load_config(path)
        apply_cli_overrides(config, parse(["x.png", "--gpu-id", "0"]))
        assert config.pipeline.ocr.gpu_id == 0

    def test_out_dir_override(self, isolated: Path) -> None:
        config = load_config()
        apply_cli_overrides(config, parse(["x.png", "-o", "myout"]))
        assert config.output.out_dir == "myout"

    def test_ocr_dir_overrides(self, isolated: Path) -> None:
        config = load_config()
        apply_cli_overrides(
            config, parse(["x.png", "--det-dir", "/d", "--rec-dir", "/r"])
        )
        assert config.pipeline.ocr.det_model_dir == "/d"
        assert config.pipeline.ocr.rec_model_dir == "/r"

    def test_all_flag_defaults_are_none(self) -> None:
        """기본값이 None 이어야 '미지정'과 'false 로 지정'을 구분할 수 있다."""
        args = parse(["x.png"])
        for name in (
            "out", "image", "font", "show_ocr_boxes", "include_ocr",
            "deskew", "long_side", "tiles", "tile_overlap", "hint",
            "samples", "sample_temperature",
            "crop_pad", "crop_upscale", "geometry_fallback",
            "det_dir", "rec_dir", "cls_dir", "gpu_id",
            "base_url", "model", "image_max_side",
        ):
            assert getattr(args, name) is None, f"--{name} 의 기본값이 None 이 아닙니다"


class TestAppConfigDefaults:
    def test_constructible_without_arguments(self) -> None:
        config = AppConfig()
        assert config.pipeline.detect.tiles == 3
        assert config.output.out_dir == "out"
        assert config.source_path is None
