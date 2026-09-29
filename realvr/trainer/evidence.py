"""Evidence supervision for ReaLVR (Section 3 of the paper).

Notation follows the paper:
    K       number of latent positions
    z_t     regenerated latent state at position t
    p+      positive visual prototype (ROI-masked mean of the visual tokens)
    N       set of negative prototypes pooled from other examples
    g_t     sim(z_t, p+) - max_{p- in N} sim(z_t, p-)
    r_t(y)  attention readout of latent position t under candidate answer y
    gamma_t [r_t(y*) - mean_{y- in Y-} r_t(y-)]_+
    w_t     eta / K + (1 - eta) * gamma_t
    l_ev    sum_t sg(w_t) * [m_ev - g_t]_+
"""

import contextlib
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------

def distinct_wrong_answers(answers: Sequence[Optional[str]], correct: Sequence[bool]) -> List[str]:
    """Distinct parseable wrong answers of one prompt group, in order of first appearance."""
    seen, wrong = set(), []
    for answer, is_correct in zip(answers, correct):
        if answer is None or is_correct:
            continue
        key = answer.casefold()
        if key not in seen:
            seen.add(key)
            wrong.append(answer)
    return wrong


def format_answer(tokenizer, answer: str) -> Tuple[List[int], List[bool]]:
    """Tokenize Fmt(y) = <answer>y</answer> and mark the content tokens J(y).

    A token belongs to J(y) when its character span overlaps the answer content,
    so the markers are excluded even when BPE merges across the boundary.
    """
    prefix, suffix = "<answer>", "</answer>"
    text = prefix + answer + suffix
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = len(prefix), len(prefix) + len(answer)
    content = [s < end and e > start for s, e in encoding["offset_mapping"]]
    return list(encoding["input_ids"]), content


# ---------------------------------------------------------------------------
# Visual prototypes
# ---------------------------------------------------------------------------

def box_token_indices(box: Sequence[float], grid_h: int, grid_w: int) -> torch.Tensor:
    """Indices of the visual tokens covered by a normalized [x1, y1, x2, y2] box on a grid_h x grid_w grid."""
    x1, y1, x2, y2 = (min(max(float(v), 0.0), 1.0) for v in box[:4])
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    c1 = min(int(x1 * grid_w), grid_w - 1)
    r1 = min(int(y1 * grid_h), grid_h - 1)
    c2 = max(min(math.ceil(x2 * grid_w), grid_w), c1 + 1)
    r2 = max(min(math.ceil(y2 * grid_h), grid_h), r1 + 1)
    rows = torch.arange(r1, r2)
    cols = torch.arange(c1, c2)
    return (rows[:, None] * grid_w + cols[None, :]).flatten()


def region_prototype(
    image_embeds: Sequence[torch.Tensor],
    grids: Sequence[Tuple[int, int]],
    boxes: Sequence[Tuple[int, Sequence[float]]],
) -> torch.Tensor:
    """Positive prototype p+ of one example.

    image_embeds: per image, the merged visual tokens [n_i, H] in raster order.
    grids:        per image, the merged token grid (h_i, w_i) with h_i * w_i = n_i.
    boxes:        (image index, normalized box) pairs from the ROI annotation.
    Falls back to the whole-image mean when no box selects any token.
    """
    selected = []
    for image_idx, box in boxes:
        if not 0 <= image_idx < len(image_embeds):
            continue
        h, w = grids[image_idx]
        idx = box_token_indices(box, h, w).to(image_embeds[image_idx].device)
        selected.append(image_embeds[image_idx].index_select(0, idx))
    if selected:
        return torch.cat(selected, dim=0).float().mean(dim=0)
    return torch.cat(list(image_embeds), dim=0).float().mean(dim=0)


def sample_negatives(
    prototypes: torch.Tensor,
    groups: torch.Tensor,
    group: int,
    num_negatives: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Up to `num_negatives` nonzero prototypes from examples outside `group`.

    prototypes: [M, H] prototypes of every example in the global batch.
    groups:     [M] prompt-group id of each prototype.
    """
    valid = (groups != group) & (prototypes.norm(dim=-1) > 0)
    candidates = torch.nonzero(valid.cpu(), as_tuple=False).flatten()
    if candidates.numel() > num_negatives:
        order = torch.randperm(candidates.numel(), generator=generator)[:num_negatives]
        candidates = candidates[order]
    return prototypes.index_select(0, candidates.to(prototypes.device))


# ---------------------------------------------------------------------------
# Answer readout
# ---------------------------------------------------------------------------

class AnswerReadout:
    """Accumulates r_t(y): post-softmax attention from the answer-content queries to the latent keys,
    averaged over decoder layers, heads, and content positions.

    rows:     [N, R] query positions of the content tokens (padded).
    row_mask: [N, R] valid entries of `rows`.
    cols:     [N, K] positions of the K latent tokens.
    """

    def __init__(self, rows: torch.Tensor, row_mask: torch.Tensor, cols: torch.Tensor):
        self.rows = rows
        self.row_mask = row_mask
        self.cols = cols
        self.total = None
        self.num_layers = 0

    @torch.no_grad()
    def record(self, module, query, key, attention_mask, scaling):
        n, num_heads, _, head_dim = query.shape
        key_len = key.shape[2]
        rows = self.rows.to(query.device)
        cols = self.cols.to(query.device)
        if key.shape[1] != num_heads:
            key = key.repeat_interleave(num_heads // key.shape[1], dim=1)
        q = query.gather(2, rows[:, None, :, None].expand(-1, num_heads, -1, head_dim))
        scores = torch.matmul(q, key.transpose(-1, -2)).float() * scaling  # [N, H, R, L]

        allowed = torch.arange(key_len, device=query.device)[None, None, None, :] <= rows[:, None, :, None]
        if attention_mask is not None:
            mask = attention_mask[..., :key_len]
            mask = mask.expand(n, -1, -1, -1).gather(2, rows[:, None, :, None].expand(-1, mask.shape[1], -1, key_len))
            if mask.dtype == torch.bool:
                allowed = allowed & mask
            else:
                scores = scores + mask.float()
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        latent = probs.gather(-1, cols[:, None, None, :].expand(-1, num_heads, probs.shape[2], -1))
        latent = latent.mean(dim=1)  # [N, R, K]
        self.total = latent if self.total is None else self.total + latent
        self.num_layers += 1

    def readout(self) -> torch.Tensor:
        """[N, K] readout averaged over layers and valid content positions."""
        per_row = self.total / max(self.num_layers, 1)
        weight = self.row_mask.to(per_row.device, per_row.dtype)
        return (per_row * weight[..., None]).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)[:, None]


@contextlib.contextmanager
def record_readout(language_model, readout: AnswerReadout):
    """Route the decoder self-attention through SDPA and record `readout` from every layer.

    The attention output is computed by the standard SDPA kernel, so the forward pass is unchanged;
    only the rows needed for the readout are recomputed explicitly.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attention_modules = [layer.self_attn for layer in language_model.layers]
    tracked = {id(module) for module in attention_modules}
    config = attention_modules[0].config
    previous_impl = config._attn_implementation
    previous_local = ALL_ATTENTION_FUNCTIONS._local_mapping.get("sdpa")
    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa_with_readout(module, query, key, value, attention_mask, **kwargs):
        output = sdpa(module, query, key, value, attention_mask, **kwargs)
        if id(module) in tracked:
            scaling = kwargs.get("scaling")
            readout.record(module, query, key, attention_mask, module.scaling if scaling is None else scaling)
        return output

    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_with_readout
    config._attn_implementation = "sdpa"
    try:
        yield readout
    finally:
        config._attn_implementation = previous_impl
        if previous_local is None:
            del ALL_ATTENTION_FUNCTIONS["sdpa"]
        else:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = previous_local


# ---------------------------------------------------------------------------
# Weights and loss
# ---------------------------------------------------------------------------

def credit_weights(r_pos: torch.Tensor, r_neg: torch.Tensor, eta: float) -> torch.Tensor:
    """w_t = eta / K + (1 - eta) * [r+_t - r-_t]_+, detached. Inputs are [B, K]."""
    gamma = (r_pos - r_neg).clamp_min(0.0)
    num_latents = r_pos.shape[-1]
    return (eta / num_latents + (1.0 - eta) * gamma).detach()


def evidence_loss(
    latents: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    weights: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """l_ev = sum_t sg(w_t) * [m_ev - g_t]_+ for one example.

    latents:   [K, H] regenerated latent states (with gradient).
    positive:  [H] positive prototype p+.
    negatives: [n, H] negative prototypes, n >= 1.
    weights:   [K] routing weights w_t.
    """
    z = latents.float()
    sim_pos = F.cosine_similarity(z, positive.float()[None, :], dim=-1)
    sim_neg = F.cosine_similarity(z[:, None, :], negatives.float()[None, :, :], dim=-1).max(dim=1).values
    gap = sim_pos - sim_neg
    return (weights.detach().float() * (margin - gap).clamp_min(0.0)).sum()
