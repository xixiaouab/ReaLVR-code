"""Latent visual reasoning on Qwen2.5-VL.

A latent step feeds the final-layer hidden state of the previous position back as the input
embedding of the next position. Generation enters latent mode after emitting <|lvr_start|>,
runs a fixed number of latent steps, and then continues decoding text.

Adapted from LVR (https://github.com/VincentLeebang/lvr, Apache-2.0) with modifications.
"""

import inspect
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import GenerationConfig, Qwen2_5_VLForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import ModelOutput

from realvr.constants import IGNORE_INDEX


@dataclass
class LVROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    loss_lvr: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    last_position_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class LVRGenerateOutput(GenerateDecoderOnlyOutput):
    """Generated sequences plus the latent states that were fed at the latent positions.

    latent_states: [B, K, H]; latent_mask: [B, K] marks the latent steps that were taken.
    """

    latent_states: Optional[torch.FloatTensor] = None
    latent_mask: Optional[torch.BoolTensor] = None


def image_features(model, pixel_values, image_grid_thw):
    """Merged visual tokens of all images in the batch, concatenated in order.

    The vision tower runs without gradient when it is frozen.
    """
    frozen = not any(p.requires_grad for p in model.visual.parameters())
    with torch.no_grad() if frozen else torch.enable_grad():
        embeds = model.model.get_image_features(pixel_values, image_grid_thw)
    embeds = torch.cat(embeds, dim=0)
    return embeds.detach() if frozen else embeds


def _insert_image_features(model, input_ids, inputs_embeds, pixel_values, image_grid_thw):
    embeds = image_features(model, pixel_values, image_grid_thw).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask = input_ids == model.config.image_token_id
    if int(image_mask.sum()) != embeds.shape[0]:
        raise ValueError(f"Image features and image tokens do not match: {int(image_mask.sum())} vs {embeds.shape[0]}")
    return inputs_embeds.masked_scatter(image_mask[..., None].expand_as(inputs_embeds), embeds), embeds


def lvr_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    lvr_mode_switch: Optional[torch.BoolTensor] = None,
    last_position_hidden_state: Optional[torch.FloatTensor] = None,
    lvr_states: Optional[torch.FloatTensor] = None,
    lvr_mask: Optional[torch.BoolTensor] = None,
    logits_to_keep: Optional[int] = None,
    **kwargs,
) -> LVROutput:
    """Forward pass used by Stage 2 and at inference.

    Generation: rows with `lvr_mode_switch` take `last_position_hidden_state` as the input
    embedding of their newest position.
    Teacher forcing: positions marked by `lvr_mask` [B, L] take `lvr_states` [B, L, H].
    """
    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
    if lvr_mode_switch is not None and last_position_hidden_state is not None and bool(lvr_mode_switch.any()):
        switch = lvr_mode_switch.to(inputs_embeds.device).view(-1, 1)
        newest = torch.where(switch, last_position_hidden_state.to(inputs_embeds.dtype), inputs_embeds[:, -1])
        inputs_embeds = torch.cat([inputs_embeds[:, :-1], newest[:, None]], dim=1)
    if lvr_states is not None:
        mask = lvr_mask.to(inputs_embeds.device)[..., None]
        inputs_embeds = torch.where(mask, lvr_states.to(inputs_embeds.dtype), inputs_embeds)
    if pixel_values is not None:
        inputs_embeds, _ = _insert_image_features(self, input_ids, inputs_embeds, pixel_values, image_grid_thw)

    outputs = self.model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        image_grid_thw=image_grid_thw,
        cache_position=cache_position,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    kept = hidden_states[:, -int(logits_to_keep):] if logits_to_keep else hidden_states
    return LVROutput(
        logits=self.lm_head(kept),
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        last_position_hidden_state=hidden_states[:, -1],
    )


def lvr_sft_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    lvr_tokens: Optional[List[torch.LongTensor]] = None,
    **kwargs,
) -> LVROutput:
    """Stage 1: target-conditioned visual supervision.

    Each <|lvr|> placeholder takes the visual token of the ROI it stands for (`lvr_tokens` holds,
    per example, the indices of those tokens within the example's visual tokens). The hidden state
    one position earlier is trained to reconstruct that visual token (MSE), alongside the
    next-token loss on the text.
    """
    inputs_embeds = self.model.get_input_embeddings()(input_ids)
    lvr_positions = torch.nonzero(input_ids == self.config.lvr_id, as_tuple=True)
    targets = None
    if pixel_values is not None:
        inputs_embeds, visual = _insert_image_features(self, input_ids, inputs_embeds, pixel_values, image_grid_thw)
        if lvr_tokens is not None and lvr_positions[0].numel() > 0:
            counts = (input_ids == self.config.image_token_id).sum(dim=1)
            offsets = torch.cumsum(nn.functional.pad(counts, (1, 0)), dim=0)[:-1]
            index = torch.cat([ids.to(visual.device) + offsets[b] for b, ids in enumerate(lvr_tokens)])
            targets = visual.index_select(0, index)
            if targets.shape[0] != lvr_positions[0].numel():
                raise ValueError(f"{lvr_positions[0].numel()} <|lvr|> tokens but {targets.shape[0]} ROI tokens")
            inputs_embeds = inputs_embeds.index_put(lvr_positions, targets.to(inputs_embeds.dtype))
    else:
        # Keep every rank's vision tower in the graph for text-only batches (required by ZeRO-3).
        dummy_pixels = torch.zeros(784, 1176, device=inputs_embeds.device, dtype=self.visual.dtype)
        dummy_grid = torch.tensor([[1, 28, 28]], device=inputs_embeds.device)
        inputs_embeds = inputs_embeds + self.visual(dummy_pixels, grid_thw=dummy_grid).mean() * 0

    outputs = self.model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    logits = self.lm_head(hidden_states)

    loss_ce = loss_lvr = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].float().reshape(-1, logits.shape[-1])
        shift_labels = labels[..., 1:].masked_fill(labels[..., 1:] == self.config.lvr_id, IGNORE_INDEX)
        loss_ce = CrossEntropyLoss()(shift_logits, shift_labels.reshape(-1).to(shift_logits.device))
        if targets is not None:
            predicted = hidden_states[lvr_positions[0], lvr_positions[1] - 1].float()
            loss_lvr = MSELoss()(predicted, targets.float())
    return LVROutput(loss_ce=loss_ce, loss_lvr=loss_lvr, logits=logits, rope_deltas=outputs.rope_deltas)


class QwenWithLVR(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL with latent visual reasoning (Stage 2 and inference)."""

    forward = lvr_forward

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        lvr_steps: Optional[int] = None,
        force_lvr_start: Optional[bool] = None,
        **kwargs,
    ):
        """Sampling (or greedy) decoding with a fixed number of latent steps after each <|lvr_start|>.

        lvr_steps / force_lvr_start default to the attributes of the same name on `generation_config`.
        With `return_dict_in_generate=True` the latent states that were fed are returned as well.
        """
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, None, **kwargs)
        if lvr_steps is None:
            lvr_steps = int(getattr(generation_config, "lvr_steps", 8))
        if force_lvr_start is None:
            force_lvr_start = bool(getattr(generation_config, "force_lvr_start", False))
        if synced_gpus is None:
            synced_gpus = is_deepspeed_zero3_enabled() and dist.is_initialized() and dist.get_world_size() > 1

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, model_kwargs.get("attention_mask") is not None, device=device)
        if model_kwargs.get("attention_mask") is None:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=kwargs.get("max_length") is None and generation_config.max_length is not None,
            has_default_min_length=kwargs.get("min_length") is None and generation_config.min_length is not None,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids.shape[-1],
        )
        cache_args = [generation_config, model_kwargs, None, input_ids.shape[0], generation_config.max_length - 1, device]
        num_cache_args = len(
            [p for p in inspect.signature(self._prepare_cache_for_generation).parameters.values()
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        )
        self._prepare_cache_for_generation(*cache_args[:num_cache_args])
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor if logits_processor is not None else LogitsProcessorList(),
            device=device,
            model_kwargs=model_kwargs,
        )
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(),
        )
        model_kwargs["use_cache"] = generation_config.use_cache
        return self._decode_with_latent_steps(
            input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus,
            lvr_steps, force_lvr_start, **model_kwargs,
        )

    def _decode_with_latent_steps(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        lvr_steps: int,
        force_lvr_start: bool,
        **model_kwargs,
    ):
        """Token-by-token decoding.

        After a step whose input token is <|lvr_start|>, the next `lvr_steps` steps are latent:
        the token sampled at each of them is kept in the sequence, but its input embedding is
        replaced by the final hidden state of the previous position. Only the first <|lvr_start|>
        of a sequence opens a latent span.
        """
        pad_token_id = generation_config._pad_token_tensor
        has_eos_stopping_criteria = any(hasattr(c, "eos_token_id") for c in stopping_criteria)
        batch_size, cur_len = input_ids.shape
        device = input_ids.device
        unfinished = torch.ones(batch_size, dtype=torch.long, device=device)
        this_peer_finished = False
        model_kwargs = self._get_initial_cache_position(cur_len, device, model_kwargs)

        in_latent = torch.zeros(batch_size, dtype=torch.bool, device=device)
        span_used = torch.zeros(batch_size, dtype=torch.bool, device=device)
        remaining = torch.zeros(batch_size, dtype=torch.long, device=device)
        last_hidden = None
        latent_states = None
        latent_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        pending_start = torch.full((batch_size,), bool(force_lvr_start), dtype=torch.bool, device=device)
        pending_start &= input_ids[:, -1] != self.config.lvr_start_id

        # Multimodal RoPE positions: computed for the prompt, then advanced by one per generated token.
        # (Tracking them here keeps left-padded rows identical to decoding them alone.)
        position_ids, _ = self.model.get_rope_index(
            input_ids,
            model_kwargs.get("image_grid_thw"),
            model_kwargs.get("video_grid_thw"),
            attention_mask=model_kwargs.get("attention_mask"),
        )
        next_position = position_ids.amax(dim=(0, 2)) + 1
        keep_scores = generation_config.return_dict_in_generate and generation_config.output_scores
        all_scores = () if keep_scores else None

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if position_ids is not None:
                model_inputs["position_ids"] = position_ids
                position_ids = None
            else:
                model_inputs["position_ids"] = next_position.view(1, -1, 1).expand(3, -1, 1)
                next_position = next_position + 1
            model_inputs.update(lvr_mode_switch=in_latent, last_position_hidden_state=last_hidden)
            if bool(in_latent.any()):
                if latent_states is None:
                    latent_states = torch.zeros(batch_size, lvr_steps, last_hidden.shape[-1],
                                                device=device, dtype=last_hidden.dtype)
                rows = torch.nonzero(in_latent, as_tuple=True)[0]
                latent_states[rows, latent_count[rows]] = last_hidden[rows]
                latent_count = latent_count + in_latent.long()
            outputs = self(**model_inputs, return_dict=True)
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)
            if synced_gpus and this_peer_finished:
                continue

            scores = logits_processor(input_ids, outputs.logits[:, -1, :].float())
            if keep_scores:
                all_scores += (scores,)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(nn.functional.softmax(scores, dim=-1), num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(scores, dim=-1)
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)
            if bool(pending_start.any()):
                force_now = pending_start & unfinished.bool()
                next_tokens = torch.where(force_now, torch.full_like(next_tokens, self.config.lvr_start_id), next_tokens)
                pending_start &= ~force_now

            entering = (~in_latent) & (~span_used) & (input_ids[:, -1] == self.config.lvr_start_id)
            span_used |= entering
            remaining = torch.where(entering, torch.full_like(remaining, lvr_steps), remaining - in_latent.long())
            in_latent = (in_latent | entering) & (remaining > 0)
            last_hidden = outputs.last_position_hidden_state

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished = (in_latent | (unfinished.bool() & ~stopping_criteria(input_ids, None))).long()
            this_peer_finished = int(unfinished.max()) == 0
            del outputs

        if not generation_config.return_dict_in_generate:
            return input_ids
        if latent_states is None:
            hidden_size = self.config.text_config.hidden_size
            latent_states = torch.zeros(batch_size, lvr_steps, hidden_size, device=device, dtype=self.dtype)
        latent_mask = torch.arange(lvr_steps, device=device)[None, :] < latent_count[:, None]
        return LVRGenerateOutput(
            sequences=input_ids, scores=all_scores, latent_states=latent_states, latent_mask=latent_mask
        )


class QwenWithLVRForSFT(QwenWithLVR):
    """Qwen2.5-VL with the Stage-1 training forward."""

    forward = lvr_sft_forward
