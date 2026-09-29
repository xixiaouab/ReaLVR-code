"""Command-line arguments."""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments
from trl import GRPOConfig


@dataclass
class ModelArguments:
    model_id: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct", metadata={"help": "Model name on the Hub or local path."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={
            "help": "Training data: a LLaVA-format JSON file, or a JSON list of "
            '{"ds_name", "data_path", "image_folder"} entries, one per dataset.'
        }
    )
    image_folder: Optional[str] = field(default=None, metadata={"help": "Folder that image paths are relative to."})
    image_min_pixels: int = field(
        default=3136, metadata={"help": "Minimum pixels per image (28 x 28 per visual token)."}
    )
    image_max_pixels: int = field(
        default=12845056, metadata={"help": "Maximum pixels per image (28 x 28 per visual token)."}
    )


@dataclass
class SFTArguments(TrainingArguments):
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Packs per GPU per step."})
    loss_lvr_lambda: float = field(default=0.1, metadata={"help": "Weight of the visual reconstruction (MSE) loss."})
    max_packed_tokens: int = field(default=16384, metadata={"help": "Token budget of one pack."})
    long_seq_threshold: int = field(
        default=4096,
        metadata={"help": "Examples this long form a pack on their own and are cut to max_packed_tokens."},
    )
    max_instance_per_batch: int = field(default=4, metadata={"help": "Maximum number of examples in one pack."})


@dataclass
class ReaLVRConfig(GRPOConfig):
    """Stage-2 configuration. Defaults follow the paper's Stage-2 recipe."""

    lvr_steps: int = field(default=8, metadata={"help": "Latent length K (training and inference)."})
    evidence_weight: float = field(default=0.2, metadata={"help": "lambda_ev; 0 disables the evidence loss."})
    evidence_margin: float = field(default=0.5, metadata={"help": "Target margin m_ev."})
    credit_eta: float = field(default=0.3, metadata={"help": "Uniform baseline eta of the credit weights."})
    num_negatives: int = field(default=16, metadata={"help": "Negative prototypes per example."})
    freeze_vision_tower: bool = field(default=True)
    freeze_merger: bool = field(default=True)

    # GRPO settings of the paper
    beta: float = field(default=0.0)
    num_generations: int = field(default=8)
    temperature: float = field(default=0.6)
    top_p: float = field(default=1.0)
    top_k: int = field(default=0)
    max_completion_length: int = field(default=192)
    learning_rate: float = field(default=5e-7)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.1)
    max_steps: int = field(default=100)
    save_steps: float = field(default=25)
