"""Stage 1: supervised fine-tuning with latent visual reconstruction.

The language model is trained with the next-token loss plus a weighted MSE loss that reconstructs
the ROI visual tokens; the vision encoder and the vision-language merger stay frozen.
"""

import torch
from transformers import AutoProcessor, HfArgumentParser

from realvr.constants import LVR_END_TOKEN, LVR_SPECIAL_TOKENS, LVR_START_TOKEN, LVR_TOKEN
from realvr.dataset import make_sft_data_module
from realvr.model.qwen_lvr import QwenWithLVRForSFT
from realvr.params import DataArguments, ModelArguments, SFTArguments
from realvr.trainer import LVRSFTTrainer


def add_lvr_tokens(processor, model):
    """Add the LVR special tokens to the tokenizer and record their ids in the model config."""
    tokenizer = processor.tokenizer
    tokenizer.add_tokens(LVR_SPECIAL_TOKENS, special_tokens=True)
    model.config.lvr_start_id = tokenizer.convert_tokens_to_ids(LVR_START_TOKEN)
    model.config.lvr_id = tokenizer.convert_tokens_to_ids(LVR_TOKEN)
    model.config.lvr_end_id = tokenizer.convert_tokens_to_ids(LVR_END_TOKEN)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    model = QwenWithLVRForSFT.from_pretrained(model_args.model_id, torch_dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    model.visual.requires_grad_(False)  # vision encoder and merger

    processor = AutoProcessor.from_pretrained(
        model_args.model_id, min_pixels=data_args.image_min_pixels, max_pixels=data_args.image_max_pixels
    )
    add_lvr_tokens(processor, model)

    trainer = LVRSFTTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        **make_sft_data_module(processor, data_args, training_args),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
