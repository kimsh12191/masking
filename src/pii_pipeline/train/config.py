"""LoRA 학습 설정.

``GroundingConfig`` (데이터 생성)와 여기 ``TrainConfig`` (학습)가 설정 파일의
``grounding`` / ``train`` 섹션에 각각 대응한다. 추론 설정과 **같은 파일**에 두는
이유는 둘이 결합되어 있기 때문이다 — 학습 데이터를 서빙과 다른 캔버스·타일로
만들면 틀린 좌표를 학습시키는데, 에러도 안 나고 결과만 나빠진다. 같은 파일에
있어야 그 결합이 눈에 보인다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """``scripts/train_grounding.py`` 설정. CLI 플래그가 이 값을 덮는다.

    Attributes:
        model: 베이스 모델. **서빙할 것과 같아야 한다** — 다르면 어댑터가 안 맞는다.
        epochs: 에폭 수.
        lr: 학습률.
        batch: 장치당 배치 크기. 이미지가 커서 1 을 넘기기 어렵다.
        grad_accum: 그래디언트 누적. 실효 배치는 ``batch × grad_accum``.
        rank: LoRA rank. 좌표 교정은 큰 용량이 필요한 과제가 아니다.
        alpha: LoRA alpha. 관례상 ``rank`` 의 2배.
        dropout: LoRA dropout.
        max_len: 토큰 상한. 넘으면 뒤가 잘리고 그 항목의 좌표를 못 배운다
            (경고가 나온다). 타일당 항목이 많으면 올린다.
        dtype: ``bf16`` / ``fp16`` / ``fp32``.
        merger_module: 전체 학습할 비전 merger 모듈 이름. **32px 토큰 격자
            아래의 정밀도가 여기서 결정된다.** 모델 구현마다 이름이 다르므로
            못 찾으면 죽으면서 ``--list-modules`` 를 안내한다.
        train_merger: merger 를 열지. 끄면 LLM LoRA 만 남는다 (A/B 용).
        vision_blocks: 비전 인코더(ViT)의 **상위 N개 블록**에 LoRA. 0 이면 동결.

            **기본 0 인 것은 순서 때문이다.** merger 만 열고 먼저 재야 어디까지가
            merger 몫인지 안다. 한꺼번에 열면 좋아져도 나빠져도 원인을 못 가른다.
        vision_prefix: 비전 타워 모듈 접두사. Qwen 은 ``visual``.
        workers: 데이터로더 워커 수 (이미지 전처리).
        seed: 학습 시드.
    """

    model: str = "Qwen/Qwen3.5-9B"
    epochs: float = 1.0
    lr: float = 1e-4
    batch: int = 1
    grad_accum: int = 8
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    max_len: int = 8192
    dtype: str = "bf16"
    merger_module: str = "visual.merger"
    train_merger: bool = True
    vision_blocks: int = 0
    vision_prefix: str = "visual"
    workers: int = 4
    seed: int = 0

    #: LoRA 를 걸 LLM 선형층. Qwen 계열 트랜스포머 블록의 표준 이름이다.
    #: ViT 는 여기 안 걸린다 (거긴 ``qkv`` 로 합쳐져 있다) — ``vision_blocks``
    #: 가 모델에서 찾아 따로 붙인다.
    lora_targets: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
