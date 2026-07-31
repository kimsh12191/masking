"""반입용 다운로드 스크립트 테스트.

네트워크를 타지 않는 부분만 검증한다: tar 구조 탐색, 체크섬, 무결성 검증,
목록 출력, 경로 탈출 방어.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from download_models import (  # noqa: E402
    MANIFEST_NAME,
    MOBILE_DET,
    MODELS,
    ModelSpec,
    cmd_list,
    cmd_verify,
    dir_digest,
    find_model_root,
    looks_like_model_dir,
    main,
    sha256_of,
)


def make_model_dir(path: Path, flavor: str = "pdmodel") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if flavor == "pdmodel":
        (path / "inference.pdmodel").write_bytes(b"model")
    else:
        (path / "inference.json").write_text("{}", encoding="utf-8")
    (path / "inference.pdiparams").write_bytes(b"params")
    return path


class TestModelSpecs:
    def test_default_set_is_det_rec_cls(self) -> None:
        assert set(MODELS) == {"det", "rec", "cls"}

    def test_every_spec_has_at_least_one_url(self) -> None:
        for spec in MODELS.values():
            assert spec.urls
            assert all(u.startswith("https://") for u in spec.urls)

    def test_key_matches_dict_key(self) -> None:
        for key, spec in MODELS.items():
            assert spec.key == key

    def test_default_det_is_server_variant(self) -> None:
        """금융 서식은 좌표 정확도가 근간이므로 server 판이 기본이어야 한다."""
        assert MODELS["det"].approx_mb > 100
        assert "server" in MODELS["det"].urls[0]

    def test_mobile_det_alternative_is_small(self) -> None:
        assert MOBILE_DET.key == "det"
        assert MOBILE_DET.approx_mb < 10

    def test_korean_rec_is_v3(self) -> None:
        """한국어는 PP-OCRv3 까지만 사전학습 추론 모델이 제공된다."""
        assert "korean" in MODELS["rec"].urls[0]
        assert "v3" in MODELS["rec"].urls[0].lower()

    def test_total_is_about_122mb(self) -> None:
        total = sum(s.approx_mb for s in MODELS.values())
        assert 120 <= total <= 125


class TestLooksLikeModelDir:
    def test_pdmodel_pair(self, tmp_path: Path) -> None:
        assert looks_like_model_dir(make_model_dir(tmp_path / "m"))

    def test_json_pair(self, tmp_path: Path) -> None:
        assert looks_like_model_dir(make_model_dir(tmp_path / "m", flavor="json"))

    def test_missing_params(self, tmp_path: Path) -> None:
        d = tmp_path / "m"
        d.mkdir()
        (d / "inference.pdmodel").write_bytes(b"x")
        assert not looks_like_model_dir(d)

    def test_empty_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "m"
        d.mkdir()
        assert not looks_like_model_dir(d)


class TestFindModelRoot:
    def test_files_at_root(self, tmp_path: Path) -> None:
        root = make_model_dir(tmp_path / "x")
        assert find_model_root(root) == root

    def test_nested_one_level(self, tmp_path: Path) -> None:
        """tar 가 한 겹 감싸진 구조 (버전에 따라 다르다)."""
        extracted = tmp_path / "x"
        inner = make_model_dir(extracted / "ch_PP-OCRv4_det_server_infer")
        assert find_model_root(extracted) == inner

    def test_nested_two_levels(self, tmp_path: Path) -> None:
        extracted = tmp_path / "x"
        inner = make_model_dir(extracted / "a" / "b")
        assert find_model_root(extracted) == inner

    def test_raises_when_absent(self, tmp_path: Path) -> None:
        extracted = tmp_path / "x"
        (extracted / "docs").mkdir(parents=True)
        (extracted / "docs" / "README.txt").write_text("nope", encoding="utf-8")
        with pytest.raises(RuntimeError, match="추론 모델 파일을 찾지 못"):
            find_model_root(extracted)


class TestDigests:
    def test_sha256_matches_hashlib(self, tmp_path: Path) -> None:
        import hashlib

        f = tmp_path / "a.bin"
        f.write_bytes(b"hello world")
        assert sha256_of(f) == hashlib.sha256(b"hello world").hexdigest()

    def test_dir_digest_lists_relative_paths(self, tmp_path: Path) -> None:
        d = make_model_dir(tmp_path / "m")
        (d / "sub").mkdir()
        (d / "sub" / "extra.txt").write_text("x", encoding="utf-8")
        digest = dir_digest(d)
        assert set(digest) == {"inference.pdmodel", "inference.pdiparams", "sub/extra.txt"}

    def test_dir_digest_is_stable(self, tmp_path: Path) -> None:
        d = make_model_dir(tmp_path / "m")
        assert dir_digest(d) == dir_digest(d)

    def test_dir_digest_changes_on_edit(self, tmp_path: Path) -> None:
        d = make_model_dir(tmp_path / "m")
        before = dir_digest(d)
        (d / "inference.pdiparams").write_bytes(b"tampered")
        assert dir_digest(d) != before


@pytest.fixture
def staged(tmp_path: Path) -> Path:
    """매니페스트까지 갖춘 반입 완료 상태의 디렉터리."""
    out = tmp_path / "ocr_models"
    entries = []
    for key in ("det", "rec", "cls"):
        d = make_model_dir(out / key)
        entries.append(
            {
                "key": key,
                "title": key,
                "source_url": "https://example.invalid/x.tar",
                "archive_sha256": "0" * 64,
                "archive_mb": 1.0,
                "files": dir_digest(d),
                "note": "",
            }
        )
    (out / MANIFEST_NAME).write_text(
        json.dumps({"models": entries, "total_mb": 3.0}, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


class TestVerify:
    def test_passes_on_intact_directory(self, staged: Path) -> None:
        assert cmd_verify(staged) == 0

    def test_detects_tampered_file(self, staged: Path) -> None:
        (staged / "rec" / "inference.pdiparams").write_bytes(b"tampered")
        assert cmd_verify(staged) == 1

    def test_detects_missing_file(self, staged: Path) -> None:
        (staged / "cls" / "inference.pdmodel").unlink()
        assert cmd_verify(staged) == 1

    def test_detects_missing_directory(self, staged: Path) -> None:
        import shutil

        shutil.rmtree(staged / "det")
        assert cmd_verify(staged) == 1

    def test_missing_manifest_returns_2(self, tmp_path: Path) -> None:
        assert cmd_verify(tmp_path) == 2

    def test_extra_file_is_reported_but_not_fatal(self, staged: Path, capsys) -> None:
        (staged / "det" / "unexpected.txt").write_text("x", encoding="utf-8")
        assert cmd_verify(staged) == 0
        assert "매니페스트에 없는 파일" in capsys.readouterr().out


class TestListCommand:
    def test_prints_urls_sizes_and_wheel_commands(self, capsys) -> None:
        assert cmd_list(dict(MODELS)) == 0
        out = capsys.readouterr().out
        assert "합계" in out
        assert "122" in out
        for spec in MODELS.values():
            assert spec.urls[0] in out
        assert "paddlepaddle-gpu" in out  # 휠이 가중치보다 크다는 점을 함께 안내

    def test_mobile_det_flag_swaps_the_spec(self, capsys) -> None:
        assert main(["--list", "--mobile-det"]) == 0
        out = capsys.readouterr().out
        assert "mobile" in out
        assert "110.0MB" not in out


class TestArchiveSafety:
    def test_path_traversal_member_is_rejected(self, tmp_path: Path) -> None:
        """신뢰할 수 없는 아카이브가 디렉터리 밖으로 쓰지 못해야 한다."""
        from download_models import fetch_spec

        payload = tmp_path / "evil.txt"
        payload.write_text("pwned", encoding="utf-8")
        archive = tmp_path / "evil.tar"
        with tarfile.open(archive, "w") as tf:
            tf.add(payload, arcname="../../escaped.txt")

        spec = ModelSpec(
            key="det", title="t", approx_mb=1.0, urls=[archive.resolve().as_uri()]
        )
        with pytest.raises(RuntimeError, match="비정상 경로"):
            fetch_spec(spec, tmp_path / "out")

    def test_valid_archive_is_extracted_and_laid_out(self, tmp_path: Path) -> None:
        src = make_model_dir(tmp_path / "src" / "ch_PP-OCRv4_det_server_infer")
        archive = tmp_path / "ok.tar"
        with tarfile.open(archive, "w") as tf:
            tf.add(src, arcname=src.name)

        from download_models import fetch_spec

        out = tmp_path / "out"
        spec = ModelSpec(
            key="det", title="t", approx_mb=1.0, urls=[archive.resolve().as_uri()]
        )
        meta = fetch_spec(spec, out)

        assert looks_like_model_dir(out / "det")
        assert meta["key"] == "det"
        assert len(str(meta["archive_sha256"])) == 64
        assert set(meta["files"]) == {"inference.pdmodel", "inference.pdiparams"}  # type: ignore[arg-type]
