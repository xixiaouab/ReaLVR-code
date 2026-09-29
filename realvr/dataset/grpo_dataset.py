"""Stage-2 prompts.

Each record is LLaVA-style JSON:
    {"image": "path/or/list", "conversations": [{"from": "human", "value": "<image>\\nQuestion"},
                                                {"from": "gpt", "value": "<answer>B</answer>"}],
     "bboxes": [[x1, y1, x2, y2], ...]}            # optional ROI evidence, pixels or normalized
The i-th box refers to image min(i, num_images - 1).
"""

import json
import os
import re

from PIL import Image
from torch.utils.data import Dataset

from realvr.constants import SYSTEM_MESSAGE


def _strip_image_tokens(text: str) -> str:
    return re.sub(r"\n?<image>\n?", "", text)


class GRPODataset(Dataset):
    def __init__(self, data_path: str, image_folder: str, image_min_pixels: int, image_max_pixels: int):
        with open(data_path) as f:
            self.records = json.load(f)
        self.image_folder = image_folder
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels

    def __len__(self):
        return len(self.records)

    def _image_path(self, path: str) -> str:
        if os.path.isabs(path) or path.startswith("http") or not self.image_folder:
            return path
        return os.path.join(self.image_folder, path)

    @staticmethod
    def _normalized_box(box, width, height):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if max(x1, y1, x2, y2) > 1.0:
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
        return [x1, y1, x2, y2]

    def __getitem__(self, index):
        record = self.records[index]
        images = record.get("image", [])
        images = [images] if isinstance(images, str) else list(images)
        image_paths = [self._image_path(p) for p in images]

        content = [
            {"type": "image", "image": path, "min_pixels": self.image_min_pixels, "max_pixels": self.image_max_pixels}
            for path in image_paths
        ]
        question, answer = record["conversations"][0], record["conversations"][1]
        content.append({"type": "text", "text": _strip_image_tokens(question["value"])})
        prompt = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": content},
        ]

        boxes = []
        for i, box in enumerate(record.get("bboxes") or []):
            image_idx = min(i, len(image_paths) - 1)
            if image_idx < 0:
                break
            with Image.open(image_paths[image_idx]) as image:
                width, height = image.size
            boxes.append((image_idx, self._normalized_box(box, width, height)))

        return {
            "prompt": prompt,
            "assistant": {"role": "assistant", "content": _strip_image_tokens(answer["value"])},
            "evidence_boxes": boxes,
        }
