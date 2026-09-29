"""Special tokens and prompts, adapted from Qwen2-VL-Finetune (https://github.com/2U1/Qwen2-VL-Finetune,
Apache-2.0) with modifications."""

IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

LVR_START_TOKEN = "<|lvr_start|>"
LVR_TOKEN = "<|lvr|>"
LVR_END_TOKEN = "<|lvr_end|>"
LVR_PLACEHOLDER = "<lvr>"
LVR_SPECIAL_TOKENS = [LVR_START_TOKEN, LVR_TOKEN, LVR_END_TOKEN]

SYSTEM_MESSAGE = "You are a helpful assistant."

MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]
