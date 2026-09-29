"""A randomly initialized, very small Qwen2.5-VL used by the CPU tests."""

import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

VOCAB = 160
IMAGE_TOKEN_ID = 150
VISION_START_ID = 151
VISION_END_ID = 152
LVR_START_ID = 153
LVR_ID = 154
LVR_END_ID = 155


def tiny_config(attn_implementation="sdpa"):
    config = Qwen2_5_VLConfig(
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            window_size=56,
            fullatt_block_indexes=[0],
        ),
        text_config=dict(
            vocab_size=VOCAB,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        ),
        vocab_size=VOCAB,
        image_token_id=IMAGE_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
    )
    config.lvr_start_id = LVR_START_ID
    config.lvr_id = LVR_ID
    config.lvr_end_id = LVR_END_ID
    config._attn_implementation = attn_implementation
    config.text_config._attn_implementation = attn_implementation
    config.vision_config._attn_implementation = attn_implementation
    return config


def tiny_qwen2_5_vl(attn_implementation="sdpa", model_class=Qwen2_5_VLForConditionalGeneration):
    torch.manual_seed(0)
    return model_class(tiny_config(attn_implementation)).float()
