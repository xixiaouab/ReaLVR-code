"""Stage 2: GRPO on free-running latent trajectories with ReaLVR evidence supervision.

Per step:
  1. Sample G completions per prompt with the current policy; the latent states fed during
     sampling are kept.
  2. GRPO loss on the answer tokens, replaying each completion with its saved latents fixed.
  3. Evidence loss (Section 3): regenerate the K latents from the prompt with gradient, weight each
     position by the answer-contrastive readout, and push it towards the ROI prototype and away
     from the closest prototype of another example.

The GRPO part follows TRL's GRPOTrainer (https://github.com/huggingface/trl, Apache-2.0), with
modifications.
"""

from collections import defaultdict

import torch
from accelerate.utils import gather, gather_object, set_seed
from qwen_vl_utils import process_vision_info
from transformers import GenerationConfig, Trainer
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import RepeatSampler
from trl.trainer.utils import selective_log_softmax

from realvr.answers import parse_answer, reference_answer
from realvr.model.qwen_lvr import image_features
from realvr.trainer.evidence import (
    AnswerReadout,
    credit_weights,
    distinct_wrong_answers,
    evidence_loss,
    format_answer,
    record_readout,
    region_prototype,
    sample_negatives,
)


class ReaLVRTrainer(Trainer):
    def __init__(self, model, reward_funcs, args, train_dataset, processing_class, callbacks=None,
                 optimizers=(None, None)):
        if args.beta != 0.0:
            raise ValueError("ReaLVR Stage 2 has no KL term; set beta = 0.")
        if args.num_iterations != 1:
            raise ValueError("ReaLVR Stage 2 takes one policy update per batch of rollouts (num_iterations = 1).")
        self.reward_funcs = list(reward_funcs)
        self.reward_names = [func.__name__ for func in self.reward_funcs]
        self.reward_weights = torch.tensor(args.reward_weights or [1.0] * len(self.reward_funcs), dtype=torch.float32)
        self.num_generations = args.num_generations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self._metrics = defaultdict(list)
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=lambda features: features,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # compute_loss returns a per-batch mean, so the Trainer has to scale it for gradient accumulation.
        self.model_accepts_loss_kwargs = False
        global_batch = args.per_device_train_batch_size * self.accelerator.num_processes
        if global_batch % self.num_generations != 0:
            raise ValueError(f"Global batch {global_batch} is not divisible by num_generations {self.num_generations}.")
        set_seed(args.seed, device_specific=True)
        tokenizer = processing_class.tokenizer
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            use_cache=True,
        )

    # ------------------------------------------------------------------ data

    def _get_train_sampler(self, train_dataset=None):
        """Each prompt appears num_generations times in a row, so a group spans consecutive ranks."""
        global_batch = self.args.per_device_train_batch_size * self.accelerator.num_processes
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=global_batch // self.num_generations,
            shuffle=self.args.shuffle_dataset,
            seed=self.args.seed,
        )

    # ------------------------------------------------------------- rollouts

    def _generate_and_score(self, inputs):
        device = self.accelerator.device
        processor = self.processing_class
        tokenizer = processor.tokenizer
        config = self.model.config
        num_gen = self.num_generations

        prompts = [example["prompt"] for example in inputs]
        texts = [maybe_apply_chat_template({"prompt": prompt}, processor)["prompt"] for prompt in prompts]
        images, _ = process_vision_info(prompts)
        prompt_inputs = processor(text=texts, images=images, padding=True, padding_side="left", return_tensors="pt")
        prompt_inputs = prompt_inputs.to(device)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as policy:
            was_training = policy.training
            policy.eval()
            rollout = policy.generate(**prompt_inputs, generation_config=self.generation_config,
                                      lvr_steps=self.args.lvr_steps)
            if was_training:
                policy.train()

        batch_size, prompt_len = prompt_ids.shape
        completion_ids = rollout.sequences[:, prompt_len:].clone()
        completion_len = completion_ids.shape[1]
        span = torch.zeros_like(completion_ids, dtype=torch.bool)
        latent_positions = torch.full((batch_size, self.args.lvr_steps), -1, dtype=torch.long, device=device)
        num_latents = rollout.latent_mask.sum(dim=1).tolist()
        for b in range(batch_size):
            starts = torch.nonzero(completion_ids[b] == config.lvr_start_id).flatten()
            if starts.numel() == 0 or num_latents[b] == 0:
                continue
            start = int(starts[0])
            positions = torch.arange(start + 1, min(start + 1 + num_latents[b], completion_len), device=device)
            completion_ids[b, positions] = config.lvr_id  # latent positions carry no token
            latent_positions[b, : positions.numel()] = positions + prompt_len
            # the span closes at <|lvr_end|>, allowing one extra marker token before it
            end = start + num_latents[b] + 1
            closing = torch.nonzero(completion_ids[b, end : end + 2] == config.lvr_end_id).flatten()
            span[b, start : end + (int(closing[0]) + 1 if closing.numel() else 0)] = True

        is_eos = completion_ids == tokenizer.eos_token_id
        eos_index = torch.where(is_eos.any(dim=1), is_eos.int().argmax(dim=1),
                                torch.full((batch_size,), completion_len, device=device))
        completion_mask = (torch.arange(completion_len, device=device)[None, :] <= eos_index[:, None]).long()
        if self.args.mask_truncated_completions:
            completion_mask = completion_mask * is_eos.any(dim=1, keepdim=True).long()
        loss_mask = completion_mask.bool() & ~span

        completion_texts = [tokenizer.decode(completion_ids[b, : int(eos_index[b])], skip_special_tokens=False)
                            for b in range(batch_size)]
        completions = [[{"role": "assistant", "content": text}] for text in completion_texts]
        columns = {key: [example[key] for example in inputs] for key in inputs[0] if key != "prompt"}
        rewards_per_func = torch.zeros(batch_size, len(self.reward_funcs), device=device)
        for i, func in enumerate(self.reward_funcs):
            values = func(prompts=prompts, completions=completions, **columns)
            rewards_per_func[:, i] = torch.tensor(values, dtype=torch.float32, device=device)

        # Group statistics are computed over the global batch: a prompt's G completions may sit on different ranks.
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device)).sum(dim=1)
        grouped = rewards.view(-1, num_gen)
        mean = grouped.mean(dim=1).repeat_interleave(num_gen)
        std = grouped.std(dim=1).repeat_interleave(num_gen)
        advantages = rewards - mean
        if self.args.scale_rewards:
            advantages = advantages / (std + 1e-4)
        offset = self.accelerator.process_index * batch_size
        advantages = advantages[offset : offset + batch_size]

        correct = (rewards_per_func[:, self.reward_names.index("accuracy_reward")] > 0).tolist()
        answers = gather_object([parse_answer(text) for text in completion_texts])
        groups = torch.tensor([(offset + b) // num_gen for b in range(batch_size)], device=device)
        wrong_answers = []
        for group in groups.tolist():
            members = slice(group * num_gen, (group + 1) * num_gen)
            wrong_answers.append(distinct_wrong_answers(answers[members], correct[members]))

        lengths = gather(completion_mask.sum(dim=1).float())
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(grouped.std(dim=1).mean().item())
        for i, name in enumerate(self.reward_names):
            self._metrics[f"rewards/{name}"].append(rewards_per_func[:, i].mean().item())
        self._metrics["completions/mean_length"].append(lengths.mean().item())
        self._metrics["completions/clipped_ratio"].append(1.0 - gather(is_eos.any(dim=1).float()).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "loss_mask": loss_mask,
            "advantages": advantages,
            "pixel_values": prompt_inputs["pixel_values"],
            "image_grid_thw": prompt_inputs["image_grid_thw"],
            "latent_states": rollout.latent_states.detach(),
            "latent_positions": latent_positions,
            "gold_answers": [reference_answer(example["assistant"]["content"]) for example in inputs],
            "wrong_answers": wrong_answers,
            "evidence_boxes": [example.get("evidence_boxes", []) for example in inputs],
            "groups": groups,
        }

    # ----------------------------------------------------------------- loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("ReaLVRTrainer does not return model outputs.")
        batch = self._generate_and_score(inputs)
        loss = self._policy_loss(model, batch)
        if self.args.evidence_weight > 0:
            loss = loss + self.args.evidence_weight * self._evidence_loss(model, batch)
        return loss

    def _policy_loss(self, model, batch):
        """Clipped GRPO objective on the answer tokens; latent positions replay their saved states."""
        input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)
        attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)
        lvr_states, lvr_mask = _place_latents(batch["latent_states"], batch["latent_positions"], input_ids.shape)
        completion_len = batch["completion_ids"].shape[1]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            lvr_states=lvr_states,
            lvr_mask=lvr_mask,
            logits_to_keep=completion_len + 1,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1].float() / self.args.temperature
        logps = selective_log_softmax(logits, batch["completion_ids"])
        ratio = torch.exp(logps - logps.detach())
        advantages = batch["advantages"][:, None]
        per_token = -torch.min(ratio * advantages,
                               ratio.clamp(1 - self.epsilon_low, 1 + self.epsilon_high) * advantages)
        mask = batch["loss_mask"].float()
        if self.args.loss_type == "grpo":
            return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
        if self.args.loss_type == "bnpo":
            return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
        if self.args.loss_type == "dr_grpo":
            return (per_token * mask).sum() / (per_token.shape[0] * self.args.max_completion_length)
        raise ValueError(f"Unknown loss_type {self.args.loss_type}")

    def _evidence_loss(self, model, batch):
        """Mean over the local examples of l_ev = sum_t sg(w_t) [m_ev - g_t]_+ (Eq. realvr_batch_evidence)."""
        policy = self.accelerator.unwrap_model(model)
        latents = self._regenerate_latents(model, batch)
        r_pos, r_neg, has_wrong = self._answer_readout(model, policy, batch, latents.detach())
        weights = credit_weights(r_pos, r_neg, self.args.credit_eta)
        prototypes = self._region_prototypes(policy, batch)
        all_prototypes = gather(prototypes)
        all_groups = gather(batch["groups"])
        generator = torch.Generator().manual_seed(self.args.seed * 1_000_003 + self.state.global_step)

        losses, num_negatives = [], []
        for b in range(latents.shape[0]):
            negatives = sample_negatives(all_prototypes, all_groups, int(batch["groups"][b]),
                                         self.args.num_negatives, generator)
            num_negatives.append(negatives.shape[0])
            if negatives.shape[0] == 0 or prototypes[b].norm() == 0:
                losses.append(latents[b].sum() * 0.0)
                continue
            losses.append(evidence_loss(latents[b], prototypes[b], negatives, weights[b], self.args.evidence_margin))
        loss = torch.stack(losses).mean()

        gamma = (r_pos - r_neg).clamp_min(0.0).sum(dim=1)
        self._metrics["evidence/loss"].append(gather(loss.detach()[None]).mean().item())
        self._metrics["evidence/credit_mass"].append(gather(gamma).mean().item())
        self._metrics["evidence/wrong_answer_rate"].append(gather(has_wrong.float()).mean().item())
        self._metrics["evidence/negatives"].append(float(sum(num_negatives)) / max(len(num_negatives), 1))
        return loss

    def _regenerate_latents(self, model, batch):
        """z_t = T_theta(x, <|lvr_start|>, z_1..z_{t-1}) for t = 1..K, keeping the gradient through the recurrence."""
        config = self.model.config
        prompt_ids, prompt_mask = batch["prompt_ids"], batch["prompt_mask"]
        batch_size, prompt_len = prompt_ids.shape
        input_ids = torch.cat([prompt_ids, prompt_ids.new_full((batch_size, 1), config.lvr_start_id)], dim=1)
        attention_mask = torch.cat([prompt_mask, prompt_mask.new_ones(batch_size, 1)], dim=1)
        states = []
        for step in range(self.args.lvr_steps):
            if step > 0:
                input_ids = torch.cat([input_ids, input_ids.new_full((batch_size, 1), config.lvr_id)], dim=1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch_size, 1)], dim=1)
            lvr_states = lvr_mask = None
            if states:
                fed = torch.stack(states, dim=1)
                lvr_states = torch.cat([fed.new_zeros(batch_size, prompt_len + 1, fed.shape[-1]), fed], dim=1)
                lvr_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
                lvr_mask[:, prompt_len + 1 :] = True
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                lvr_states=lvr_states,
                lvr_mask=lvr_mask,
                logits_to_keep=1,
                use_cache=False,
            )
            states.append(outputs.last_position_hidden_state)
        return torch.stack(states, dim=1)

    @torch.no_grad()
    def _answer_readout(self, model, policy, batch, latents):
        """r+ (correct answer) and r- (mean over distinct wrong answers) for every example, [B, K] each.

        Every candidate answer is teacher-forced after the same regenerated latent span; all candidates of the
        local batch go through one forward pass.
        """
        config = self.model.config
        tokenizer = self.processing_class.tokenizer
        num_latents = self.args.lvr_steps
        batch_size = latents.shape[0]
        patches_per_image = batch["image_grid_thw"].prod(dim=-1).tolist()
        images_per_example = (batch["prompt_ids"] == config.vision_start_token_id).sum(dim=1).tolist()
        patch_offsets = [0]
        for count in patches_per_image:
            patch_offsets.append(patch_offsets[-1] + count)
        first_image = [sum(images_per_example[:b]) for b in range(batch_size)]

        sequences, content_rows, latent_cols, owners, is_gold = [], [], [], [], []
        for b in range(batch_size):
            prompt = batch["prompt_ids"][b][batch["prompt_mask"][b].bool()].tolist()
            head = prompt + [config.lvr_start_id] + [config.lvr_id] * num_latents + [config.lvr_end_id]
            gold = batch["gold_answers"][b]
            candidates = [(gold, True)] + ([(a, False) for a in batch["wrong_answers"][b]] if gold else [])
            for answer, gold_flag in candidates:
                # an example without a reference answer still contributes one row, so that every rank
                # runs the same forward pass; its readout is discarded below
                answer_ids, content = format_answer(tokenizer, answer or "-")
                sequences.append(head + answer_ids)
                content_rows.append([len(head) + j for j, keep in enumerate(content) if keep])
                latent_cols.append([len(prompt) + 1 + t for t in range(num_latents)])
                owners.append(b)
                is_gold.append(gold_flag)

        device = batch["prompt_ids"].device
        max_len = max(len(seq) for seq in sequences)
        max_rows = max(len(rows) for rows in content_rows)
        num = len(sequences)
        input_ids = torch.full((num, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(num, max_len, dtype=torch.long, device=device)
        rows = torch.zeros(num, max_rows, dtype=torch.long, device=device)
        row_mask = torch.zeros(num, max_rows, dtype=torch.bool, device=device)
        cols = torch.zeros(num, num_latents, dtype=torch.long, device=device)
        lvr_states = latents.new_zeros(num, max_len, latents.shape[-1])
        lvr_mask = torch.zeros(num, max_len, dtype=torch.bool, device=device)
        pixel_values, image_grid_thw = [], []
        for n, seq in enumerate(sequences):
            shift = max_len - len(seq)  # left padding
            input_ids[n, shift:] = torch.tensor(seq, device=device)
            attention_mask[n, shift:] = 1
            rows[n, : len(content_rows[n])] = torch.tensor(content_rows[n], device=device) + shift
            row_mask[n, : len(content_rows[n])] = True
            cols[n] = torch.tensor(latent_cols[n], device=device) + shift
            lvr_states[n, cols[n]] = latents[owners[n]]
            lvr_mask[n, cols[n]] = True
            first = first_image[owners[n]]
            last = first + images_per_example[owners[n]]
            pixel_values.append(batch["pixel_values"][patch_offsets[first] : patch_offsets[last]])
            image_grid_thw.append(batch["image_grid_thw"][first:last])

        readout = AnswerReadout(rows, row_mask, cols)
        was_training = model.training
        model.eval()
        with record_readout(policy.model.language_model, readout):
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=torch.cat(pixel_values),
                image_grid_thw=torch.cat(image_grid_thw),
                lvr_states=lvr_states,
                lvr_mask=lvr_mask,
                logits_to_keep=1,
                use_cache=False,
            )
        if was_training:
            model.train()
        values = readout.readout()

        r_pos = latents.new_zeros(batch_size, num_latents, dtype=torch.float32)
        r_neg = latents.new_zeros(batch_size, num_latents, dtype=torch.float32)
        has_wrong = torch.zeros(batch_size, dtype=torch.bool, device=device)
        owners_t = torch.tensor(owners, device=device)
        gold_t = torch.tensor(is_gold, device=device)
        for b in range(batch_size):
            gold_rows = values[(owners_t == b) & gold_t]
            wrong_rows = values[(owners_t == b) & ~gold_t]
            if not batch["gold_answers"][b]:
                continue
            r_pos[b] = gold_rows[0]
            r_neg[b] = wrong_rows.mean(dim=0) if wrong_rows.shape[0] > 0 else gold_rows[0]
            has_wrong[b] = wrong_rows.shape[0] > 0
        return r_pos, r_neg, has_wrong

    @torch.no_grad()
    def _region_prototypes(self, policy, batch):
        """p+ for every example: mean visual token inside its ROI boxes (whole image without boxes)."""
        config = self.model.config
        merge = policy.model.visual.spatial_merge_size
        grids = batch["image_grid_thw"]
        embeds = image_features(policy, batch["pixel_values"], grids)
        per_image = torch.split(embeds, (grids.prod(dim=-1) // merge**2).tolist())
        token_grids = [(int(h) // merge, int(w) // merge) for _, h, w in grids.tolist()]
        images_per_example = (batch["prompt_ids"] == config.vision_start_token_id).sum(dim=1).tolist()
        prototypes, first = [], 0
        for b, count in enumerate(images_per_example):
            prototypes.append(region_prototype(per_image[first : first + count], token_grids[first : first + count],
                                               batch["evidence_boxes"][b]))
            first += count
        return torch.stack(prototypes)

    # -------------------------------------------------------------- logging

    def log(self, logs, start_time=None):
        metrics = {key: sum(values) / len(values) for key, values in self._metrics.items() if values}
        super().log({**logs, **metrics}, start_time)
        self._metrics.clear()


def _place_latents(latent_states, latent_positions, shape):
    """Scatter per-example latent states [B, K, H] to their absolute positions in a [B, L] sequence."""
    batch_size, length = shape
    states = latent_states.new_zeros(batch_size, length, latent_states.shape[-1])
    mask = torch.zeros(batch_size, length, dtype=torch.bool, device=latent_states.device)
    valid = latent_positions >= 0
    rows = torch.arange(batch_size, device=latent_states.device)[:, None].expand_as(latent_positions)[valid]
    cols = latent_positions[valid]
    states[rows, cols] = latent_states[valid]
    mask[rows, cols] = True
    return states, mask
