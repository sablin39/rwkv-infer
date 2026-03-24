import logging
from typing import List, Optional, Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


# Dummy class so SGLang's processor registry can match by .__name__
class RWKV7VLForConditionalGeneration:
    pass


class RWKV7VLImageProcessor(BaseMultimodalProcessor):
    """SGLang multimodal processor for RWKV7VL."""

    # Link to the model class (used by SGLang's processor registry)
    models = [RWKV7VLForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.image_token_id = getattr(hf_config, "image_token_id", 65532)
        self.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 65530)
        self.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 65531)

        # The user-facing <image> token in the chat template
        self.user_image_tag = "<image>"

        # Build multimodal token spec for base class utilities
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.user_image_tag,
            image_token_id=self.image_token_id,
        )
        try:
            self.mm_tokens.build(self)
        except Exception:
            pass  # build may fail if regex patterns aren't needed

    def _find_image_token_spans(
        self,
        input_ids: List[int],
        expected_token_counts: List[int],
    ) -> List[tuple[int, int]]:
        spans = []
        idx = 0
        while idx < len(input_ids):
            if input_ids[idx] != self.vision_start_token_id:
                idx += 1
                continue

            start = idx + 1
            end = start
            while end < len(input_ids) and input_ids[end] == self.image_token_id:
                end += 1

            if end >= len(input_ids) or input_ids[end] != self.vision_end_token_id:
                raise ValueError(
                    "Malformed image token span in RWKV7VL processor output: "
                    "missing vision end token."
                )
            if end == start:
                raise ValueError(
                    "Malformed image token span in RWKV7VL processor output: "
                    "empty image placeholder region."
                )

            spans.append((start, end - 1))
            idx = end + 1

        if len(spans) != len(expected_token_counts):
            raise ValueError(
                "Number of detected image spans does not match processed images: "
                f"expected {len(expected_token_counts)}, got {len(spans)}."
            )

        for i, ((start, end), expected_count) in enumerate(zip(spans, expected_token_counts)):
            actual_count = end - start + 1
            if actual_count != expected_count:
                raise ValueError(
                    "Image token span length does not match image_grid_thw-derived "
                    f"token count for image {i}: expected {expected_count}, got {actual_count}."
                )

        return spans

    async def process_mm_data_async(
        self,
        image_data: Optional[List] = None,
        audio_data: Optional[List] = None,
        input_text: str = "",
        request_obj=None,
        **kwargs,
    ) -> dict:
        """Process multimodal inputs and return tokenized IDs + mm items."""

        if not image_data:
            # Text-only: tokenize directly
            input_ids = self._processor.tokenizer.encode(input_text)
            return {"input_ids": input_ids, "mm_items": []}

        # 1. Load images from various sources (URLs, base64, file paths, PIL)
        base_output: BaseMultiModalProcessorOutput = self.load_mm_data(
            prompt=input_text,
            multimodal_tokens=self.mm_tokens,
            image_data=image_data,
        )

        images = base_output.images
        text = base_output.input_text

        if not images:
            input_ids = self._processor.tokenizer.encode(text)
            return {"input_ids": input_ids, "mm_items": []}

        # 2. Process with HuggingFace ModRWKVProcessor
        #    This handles: <image> → <|vision_start|><|image_pad|>...<|vision_end|>
        #    expansion, image preprocessing, and tokenization.
        processed = self._processor(
            text=[text],
            images=images,
            return_tensors="pt",
        )

        input_ids = processed["input_ids"][0].tolist()

        pixel_values = processed.get("pixel_values")
        image_grid_thw = processed.get("image_grid_thw")

        if pixel_values is None or image_grid_thw is None:
            return {"input_ids": input_ids, "mm_items": []}

        # 3. Split pixel_values per image and create MultimodalDataItem objects
        #    pixel_values: [total_patches, patch_pixel_dim]
        #    image_grid_thw: [num_images, 3]
        raw_patch_counts = image_grid_thw.prod(dim=-1).tolist()
        spatial_merge_size = getattr(
            getattr(self.hf_config, "vision_config", None), "spatial_merge_size", 2
        )
        image_token_counts = (
            image_grid_thw.prod(dim=-1) // (spatial_merge_size**2)
        ).tolist()
        pixel_splits = torch.split(pixel_values, raw_patch_counts)
        image_token_spans = self._find_image_token_spans(input_ids, image_token_counts)

        mm_items = []
        for i in range(len(raw_patch_counts)):
            item = MultimodalDataItem(modality=Modality.IMAGE)
            item.feature = pixel_splits[i]
            item.image_grid_thw = image_grid_thw[i : i + 1]
            item.offsets = [image_token_spans[i]]
            mm_items.append(item)

        return {
            "input_ids": input_ids,
            "mm_items": mm_items,
            "im_token_id": self.image_token_id,
            "im_start_id": self.vision_start_token_id,
            "im_end_id": self.vision_end_token_id,
        }
