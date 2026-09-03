from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha1
import json
from typing import Literal


GraphKind = Literal["v3", "residual", "gated", "pna", "graphormer", "gps"]
StructureKind = Literal["none", "depth", "degree", "path", "combined"]
ReadoutKind = Literal["v3", "dual_attention", "virtual", "set2set", "concat"]
FusionKind = Literal["graph", "semantic", "fixed", "scalar", "gated"]
RankKind = Literal["listwise", "pairwise"]
ScheduleKind = Literal["cosine", "plateau"]
UnfreezeKind = Literal["frozen", "last2", "full"]


@dataclass
class LossConfig:
    rank_kind: RankKind = "listwise"
    use_multi_positive: bool = True
    use_twin: bool = True
    use_tool: bool = True
    use_relation: bool = True
    use_distance: bool = True
    listwise_weight: float = 1.0
    twin_weight: float = 1.0
    tool_weight: float = 0.5
    relation_weight: float = 1.0
    multi_positive_weight: float = 0.1
    distance_weight: float = 0.1
    margin: float = 0.2
    temperature: float = 0.05


@dataclass
class ModelConfig:
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    graph_kind: GraphKind = "gps"
    structure_kind: StructureKind = "combined"
    readout_kind: ReadoutKind = "dual_attention"
    fusion_kind: FusionKind = "gated"
    hidden_dim: int = 256
    embed_dim: int = 256
    graph_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.1
    rerank_k: int = 20
    rerank_layers: int = 2
    use_reranker: bool = True
    use_semantic_expert: bool = True
    unfreeze: UnfreezeKind = "last2"


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 25
    batch_size: int = 64
    lr: float = 2e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    patience: int = 6
    max_grad_norm: float = 1.0
    schedule: ScheduleKind = "cosine"
    group_aware: bool = True
    use_ema: bool = False
    use_swa: bool = False
    online_hard_negatives: bool = False
    curriculum: bool = False
    device: str = "cuda"
    num_workers: int = 0
    max_length: int = 128


@dataclass
class ExperimentConfig:
    name: str = "legr_v4"
    tier: int = 15
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def as_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        raw = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha1(raw.encode("utf-8")).hexdigest()[:10]

    def run_name(self) -> str:
        return f"{self.name}_{self.tier}t_s{self.train.seed}_{self.fingerprint()}"
