"""Stage 2: GRPO with ReaLVR evidence supervision, initialized from a Stage-1 checkpoint."""

import torch
from transformers import AutoProcessor, HfArgumentParser

from realvr.dataset.grpo_dataset import GRPODataset
from realvr.model.qwen_lvr import QwenWithLVR
from realvr.params import DataArguments, ModelArguments, ReaLVRConfig
from realvr.train.rewards import REWARD_FUNCS
from realvr.trainer.realvr_trainer import ReaLVRTrainer


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, ReaLVRConfig))
    model_args, data_args, args = parser.parse_args_into_dataclasses()

    model = QwenWithLVR.from_pretrained(
        model_args.model_id,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.visual.requires_grad_(not args.freeze_vision_tower)
    model.visual.merger.requires_grad_(not args.freeze_merger)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    processor = AutoProcessor.from_pretrained(model_args.model_id)
    dataset = GRPODataset(
        data_args.data_path,
        data_args.image_folder,
        data_args.image_min_pixels,
        data_args.image_max_pixels,
    )
    trainer = ReaLVRTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=args,
        train_dataset=dataset,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
