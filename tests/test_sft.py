"""좌표 학습 SFT 의 순수 부분 테스트.

GPU 없이 검증할 수 있는 것은 **손실 마스킹**과 **데이터 적재**뿐인데, 하필
그 둘이 조용히 틀리는 자리다.

  · 프롬프트를 손실에서 안 빼면 모델이 자기 지시문을 외운다
  · 정답 시작 위치가 한 토큰 밀리면 좌표가 통째로 밀려서 학습된다
  · 이미지가 몇 장 빠진 채로 학습이 돌면 아무도 모른다

셋 다 학습 로그로는 안 보이고 "좌표가 안 좋아지네" 로만 나타난다.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from pii_pipeline.llm.prompts import (
    SYSTEM_GROUNDING,
    SYSTEM_LOCATE,
    USER_GROUNDING,
)
from pii_pipeline.train.sft import (
    IGNORE_INDEX,
    TOKEN_AXIS_KEYS,
    Example,
    build_messages,
    check_uniform_image_size,
    load_jsonl,
    mask_prompt,
    pad_fill_value,
    select_vision_blocks,
    summarize,
    tally_image_sizes,
)


def example(findings=None, task: str = "locate", query=None) -> Example:
    items = findings if findings is not None else [
        {"text": "강동혁", "bbox_2d": [318, 266, 398, 317]}
    ]
    return Example(
        image_path=Path("tiles/x_t0.png"),
        target={"findings": items},
        meta={"tile": 0},
        task=task,
        query=query if query is not None else [i["text"] for i in items if i["text"]],
    )


# --------------------------------------------------------------------------
# 손실 마스킹
# --------------------------------------------------------------------------


class TestMaskPrompt:
    def test_prompt_tokens_are_excluded(self) -> None:
        labels = mask_prompt([10, 11, 12, 20, 21], prompt_len=3)
        assert labels == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 20, 21]

    def test_answer_tokens_are_kept_verbatim(self) -> None:
        ids = [1, 2, 3, 4, 5, 6]
        labels = mask_prompt(ids, prompt_len=2)
        assert labels[2:] == ids[2:]

    def test_length_is_preserved(self) -> None:
        """라벨과 입력 길이가 다르면 프레임워크가 조용히 잘라낸다."""
        ids = list(range(17))
        assert len(mask_prompt(ids, prompt_len=5)) == len(ids)

    def test_rejects_all_prompt(self) -> None:
        """정답 토큰이 0개면 손실이 0 이라 '학습은 도는데 안 배운다'."""
        with pytest.raises(ValueError, match="정답 토큰이 하나도 없습니다"):
            mask_prompt([1, 2, 3], prompt_len=3)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="범위를 벗어났습니다"):
            mask_prompt([1, 2, 3], prompt_len=9)
        with pytest.raises(ValueError, match="범위를 벗어났습니다"):
            mask_prompt([1, 2, 3], prompt_len=-1)

    def test_does_not_mutate_input(self) -> None:
        ids = [1, 2, 3, 4]
        mask_prompt(ids, prompt_len=2)
        assert ids == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# 배치 패딩 — 어느 텐서에 토큰 축이 있는가
# --------------------------------------------------------------------------


class TestPadding:
    def test_labels_pad_with_ignore_not_pad_token(self) -> None:
        """패딩 자리에 pad_id 를 넣으면 모델이 끝에 패딩을 뱉도록 배운다."""
        assert pad_fill_value("labels", pad_id=7) == IGNORE_INDEX

    def test_input_ids_pad_with_the_pad_token(self) -> None:
        assert pad_fill_value("input_ids", pad_id=7) == 7

    def test_attention_mask_pads_with_zero(self) -> None:
        assert pad_fill_value("attention_mask", pad_id=7) == 0

    def test_image_tensors_are_not_padded(self) -> None:
        """pixel_values 는 토큰 축이 없다. 일괄 처리하면 패치 하나만 남는다."""
        for key in ("pixel_values", "image_grid_thw"):
            assert key not in TOKEN_AXIS_KEYS
            with pytest.raises(ValueError, match="토큰 축이 없습니다"):
                pad_fill_value(key, pad_id=7)


# --------------------------------------------------------------------------
# 대화 구성
# --------------------------------------------------------------------------


class TestBuildMessages:
    def test_locate_asks_for_the_given_values(self) -> None:
        """주 과제 — 값을 알려주고 위치만 묻는다."""
        messages, _ = build_messages(example(task="locate", query=["강동혁", "서울"]))
        assert messages[0]["content"][0]["text"] == SYSTEM_LOCATE
        user = messages[1]["content"][1]["text"]
        assert "강동혁" in user and "서울" in user

    def test_read_falls_back_to_the_transcription_prompt(self) -> None:
        """도장·손글씨는 지목할 텍스트가 없어 이쪽으로만 가르칠 수 있다."""
        messages, _ = build_messages(example(task="read"))
        assert messages[0]["content"][0]["text"] == SYSTEM_GROUNDING
        assert messages[1]["content"][1]["text"] == USER_GROUNDING

    def test_has_exactly_one_image(self) -> None:
        messages, _ = build_messages(example())
        images = [
            part
            for m in messages
            for part in m["content"]
            if part.get("type") == "image"
        ]
        assert len(images) == 1

    def test_no_assistant_turn(self) -> None:
        """정답은 메시지에 넣지 않는다 — 프롬프트 길이를 정확히 재기 위해서다."""
        messages, _ = build_messages(example())
        assert [m["role"] for m in messages] == ["system", "user"]

    def test_answer_matches_the_inference_schema(self) -> None:
        _, answer = build_messages(example())
        payload = json.loads(answer)
        assert set(payload) == {"findings"}
        assert set(payload["findings"][0]) == {"text", "bbox_2d"}

    def test_answer_spacing_is_stable(self) -> None:
        """공백이 샘플마다 흔들리면 모델이 그 차이까지 배우느라 용량을 쓴다."""
        _, answer = build_messages(example())
        assert ", " not in answer and '": ' not in answer


# --------------------------------------------------------------------------
# 데이터 적재
# --------------------------------------------------------------------------


class TestLoadJsonl:
    def _write(self, tmp_path: Path, rows: list[dict], make_images: bool = True) -> Path:
        (tmp_path / "tiles").mkdir(exist_ok=True)
        if make_images:
            for row in rows:
                (tmp_path / row["image"]).write_bytes(b"fake png")
        path = tmp_path / "data.jsonl"
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )
        return path

    def test_reads_rows(self, tmp_path: Path) -> None:
        rows = [
            {"image": "tiles/a.png", "task": "read",
             "target": {"findings": []}, "meta": {"tile": 0}},
            {"image": "tiles/b.png", "task": "read",
             "target": {"findings": []}, "meta": {"tile": 1}},
        ]
        got = load_jsonl(self._write(tmp_path, rows))
        assert len(got) == 2
        assert got[0].image_path == tmp_path / "tiles/a.png"

    def test_missing_image_raises(self, tmp_path: Path) -> None:
        """조용히 건너뛰면 데이터가 반쯤 빠진 채 학습이 돈다."""
        rows = [{"image": "tiles/gone.png", "task": "read", "target": {"findings": []}}]
        path = self._write(tmp_path, rows, make_images=False)
        with pytest.raises(FileNotFoundError, match="이미지가 없습니다"):
            load_jsonl(path)

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "tiles").mkdir()
        (tmp_path / "tiles/a.png").write_bytes(b"x")
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps(
                {"image": "tiles/a.png", "task": "read", "target": {"findings": []}}
            )
            + "\n\n",
            encoding="utf-8",
        )
        assert len(load_jsonl(path)) == 1


# --------------------------------------------------------------------------
# 사전 점검
# --------------------------------------------------------------------------


class TestSummarize:
    def test_counts_items_and_textless(self) -> None:
        stats = summarize([
            example([
                {"text": "강동혁", "bbox_2d": [1, 2, 3, 4]},
                {"text": "", "bbox_2d": [5, 6, 7, 8]},
            ]),
            example([{"text": "김철수", "bbox_2d": [1, 2, 3, 4]}]),
        ])
        assert stats["samples"] == 2
        assert stats["items"] == 3
        assert stats["textless_items"] == 1

    def test_handles_empty_input(self) -> None:
        assert summarize([])["samples"] == 0


class TestCheckUniformImageSize:
    """크기가 섞인 학습 데이터를 **시작 전에** 잡는다.

    막지 않으면 첫 에폭 중간에 ``collate`` 가 죽는다. 그 실패는 DataLoader
    워커 안에서 나므로 traceback 이 두 겹이고, 메시지("Sizes of tensors must
    match except in dimension 0")에는 어느 키인지도 데이터 문제라는 단서도
    없다. 게다가 그때는 이미 9B 를 GPU 에 올린 뒤다.

    ``size_of`` 를 주입할 수 있게 만든 덕에 Pillow 없이 CPU 로 검증된다.
    """

    @staticmethod
    def sized(*sizes: tuple[int, int]):
        """크기별 샘플과 ``경로 -> 크기`` 조회 함수를 함께 만든다."""
        examples: list[Example] = []
        table: dict[Path, tuple[int, int]] = {}
        for index, wh in enumerate(sizes):
            path = Path(f"tiles/t{index}.png")
            examples.append(
                Example(image_path=path, target={"findings": []}, meta={}, query=[])
            )
            table[path] = wh
        return examples, table.__getitem__

    def test_uniform_data_passes(self) -> None:
        examples, size_of = self.sized((1760, 896), (1760, 896), (1760, 896))
        sizes = check_uniform_image_size(examples, batch=4, size_of=size_of)
        assert sizes == Counter({(1760, 896): 3})

    def test_mixed_sizes_are_blocked_when_batching(self) -> None:
        examples, size_of = self.sized((1760, 896), (1760, 1056))
        with pytest.raises(ValueError, match="균일하지 않습니다"):
            check_uniform_image_size(examples, batch=2, size_of=size_of)

    def test_error_names_the_causes_and_the_sizes(self) -> None:
        """원인 없이 raise 하면 torch 에러와 다를 게 없다."""
        examples, size_of = self.sized((1760, 896), (1760, 1056))
        with pytest.raises(ValueError) as err:
            check_uniform_image_size(examples, batch=8, size_of=size_of)
        text = str(err.value)
        assert "pipeline.canvas" in text
        assert "--batch 1" in text
        # 어떤 크기가 섞였는지 눈으로 봐야 어느 빌드가 섞였는지 짐작할 수 있다
        assert "1760x896" in text
        assert "1760x1056" in text

    def test_batch_one_warns_instead_of_raising(self) -> None:
        """이어 붙일 상대가 없어 실제로 돌아간다 — 막지 않는다.

        좌표 학습으로서는 여전히 손해지만, 그 판단은 호출자 몫이다.
        """
        examples, size_of = self.sized((1760, 896), (1760, 1056))
        sizes = check_uniform_image_size(examples, batch=1, size_of=size_of)
        assert len(sizes) == 2

    def test_long_size_lists_are_truncated(self) -> None:
        """크기가 수십 종이면 메시지가 화면을 덮는다."""
        examples, size_of = self.sized(*[(100 + i, 200) for i in range(12)])
        with pytest.raises(ValueError) as err:
            check_uniform_image_size(examples, batch=2, size_of=size_of)
        assert "그 밖에 4종" in str(err.value)

    def test_tally_counts_every_sample(self) -> None:
        examples, size_of = self.sized((10, 20), (10, 20), (30, 40))
        assert tally_image_sizes(examples, size_of) == Counter(
            {(10, 20): 2, (30, 40): 1}
        )

    def test_empty_dataset_is_not_an_error(self) -> None:
        """샘플이 없다는 것은 여기서 낼 에러가 아니다 (main 이 따로 잡는다)."""
        assert check_uniform_image_size([], batch=8) == Counter()


class TestSelectVisionBlocks:
    """ViT LoRA 대상은 **모델에서 찾는다.** 이름을 하드코딩할 수 없다.

    Qwen 계열은 LLM 이 q_proj/k_proj 인데 ViT 는 qkv 로 합쳐져 있고, MLP 이름도
    버전마다 바뀐다 (fc1/fc2 -> SwiGLU). 짐작하면 조용히 아무것도 안 걸린다.
    """

    # Qwen2-VL 풍 이름 (블록 0~3) + LLM + merger
    NAMES = (
        [f"visual.blocks.{i}.attn.qkv" for i in range(4)]
        + [f"visual.blocks.{i}.attn.proj" for i in range(4)]
        + [f"visual.blocks.{i}.mlp.fc1" for i in range(4)]
        + ["visual.patch_embed.proj", "visual.merger.mlp.0", "visual.merger.mlp.2"]
        + [f"model.layers.{i}.self_attn.q_proj" for i in range(4)]
    )

    def test_picks_only_the_top_blocks(self) -> None:
        got = select_vision_blocks(self.NAMES, "visual", 2)
        blocks = {n.split(".")[2] for n in got}
        assert blocks == {"2", "3"}

    def test_returns_full_paths(self) -> None:
        """접미사로 주면 다른 블록까지 딸려온다. 전체 경로여야 한다."""
        for name in select_vision_blocks(self.NAMES, "visual", 1):
            assert name.startswith("visual.blocks.3.")

    def test_llm_layers_are_never_included(self) -> None:
        got = select_vision_blocks(self.NAMES, "visual", 4)
        assert not any(n.startswith("model.layers") for n in got)

    def test_merger_and_patch_embed_are_excluded(self) -> None:
        """merger 는 LoRA 가 아니라 전체 학습으로 따로 다룬다."""
        got = select_vision_blocks(self.NAMES, "visual", 4)
        assert not any("merger" in n or "patch_embed" in n for n in got)

    def test_zero_blocks_means_frozen_vit(self) -> None:
        assert select_vision_blocks(self.NAMES, "visual", 0) == []

    def test_more_blocks_than_exist_is_fine(self) -> None:
        got = select_vision_blocks(self.NAMES, "visual", 99)
        assert {n.split(".")[2] for n in got} == {"0", "1", "2", "3"}

    def test_unknown_prefix_returns_nothing(self) -> None:
        """호출부가 이걸 보고 죽으며 --list-modules 를 안내한다."""
        assert select_vision_blocks(self.NAMES, "vision_tower", 2) == []

    def test_works_with_a_different_naming_scheme(self) -> None:
        """버전이 바뀌어 SwiGLU 로 가도 블록 번호만 있으면 잡힌다."""
        names = [f"visual.blocks.{i}.mlp.gate_proj" for i in range(3)]
        assert select_vision_blocks(names, "visual", 1) == ["visual.blocks.2.mlp.gate_proj"]
