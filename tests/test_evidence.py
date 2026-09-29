import torch
import torch.nn.functional as F

from realvr.answers import parse_answer
from realvr.trainer.evidence import (
    AnswerReadout,
    box_token_indices,
    credit_weights,
    distinct_wrong_answers,
    evidence_loss,
    record_readout,
    region_prototype,
    sample_negatives,
)
from tests.tiny_model import tiny_qwen2_5_vl


def test_parse_answer():
    assert parse_answer("<|lvr_start|>x<|lvr_end|><answer> B </answer>") == "B"
    assert parse_answer("<answer></answer>") is None
    assert parse_answer("<answer>A</answer><answer>B</answer>") is None
    assert parse_answer("no answer") is None


def test_distinct_wrong_answers():
    answers = ["A", "B", None, "b", "C", "A"]
    correct = [True, False, False, False, False, True]
    assert distinct_wrong_answers(answers, correct) == ["B", "C"]


def test_box_token_indices():
    idx = box_token_indices([0.0, 0.0, 0.5, 0.5], 4, 4)
    assert idx.tolist() == [0, 1, 4, 5]
    idx = box_token_indices([0.9, 0.9, 0.95, 0.95], 4, 4)
    assert idx.tolist() == [15]
    idx = box_token_indices([0.5, 0.5, 0.5, 0.5], 4, 4)
    assert idx.numel() == 1


def test_region_prototype_and_fallback():
    embeds = [torch.arange(16, dtype=torch.float32)[:, None].repeat(1, 3)]
    proto = region_prototype(embeds, [(4, 4)], [(0, [0.0, 0.0, 0.5, 0.5])])
    assert torch.allclose(proto, torch.full((3,), (0 + 1 + 4 + 5) / 4.0))
    whole = region_prototype(embeds, [(4, 4)], [])
    assert torch.allclose(whole, torch.full((3,), 7.5))


def test_sample_negatives_excludes_own_group_and_zero():
    protos = torch.randn(10, 4)
    protos[3] = 0.0
    groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    gen = torch.Generator().manual_seed(0)
    neg = sample_negatives(protos, groups, group=1, num_negatives=16, generator=gen)
    assert neg.shape[0] == 8  # 10 - own group (2 and 3); 3 is also zero
    for row in neg:
        assert not torch.allclose(row, protos[2])
    neg = sample_negatives(protos, groups, group=1, num_negatives=3, generator=gen)
    assert neg.shape[0] == 3


def test_credit_weights():
    r_pos = torch.tensor([[0.10, 0.02, 0.00, 0.04]])
    r_neg = torch.tensor([[0.05, 0.05, 0.00, 0.01]])
    w = credit_weights(r_pos, r_neg, eta=0.3)
    expected = 0.3 / 4 + 0.7 * torch.tensor([[0.05, 0.0, 0.0, 0.03]])
    assert torch.allclose(w, expected)
    assert not w.requires_grad


def test_evidence_loss_matches_formula_and_backprops():
    torch.manual_seed(0)
    z = torch.randn(4, 8, requires_grad=True)
    pos = torch.randn(8)
    negs = torch.randn(3, 8)
    w = torch.rand(4)
    loss = evidence_loss(z, pos, negs, w, margin=0.5)
    g = F.cosine_similarity(z, pos[None], dim=-1) - torch.stack(
        [F.cosine_similarity(z, n[None], dim=-1) for n in negs], dim=1).max(dim=1).values
    expected = (w * (0.5 - g).clamp_min(0)).sum()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def _reference_readout(model, input_ids, attention_mask, rows, row_mask, cols):
    """Readout from explicit eager attention maps."""
    lm = model.model.language_model
    for layer in lm.layers:
        layer.self_attn.config._attn_implementation = "eager"
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, return_dict=True)
    for layer in lm.layers:
        layer.self_attn.config._attn_implementation = "sdpa"
    maps = torch.stack([a.float() for a in out.attentions]).mean(0).mean(1)  # [N, L, L]
    result = []
    for n in range(input_ids.shape[0]):
        r = rows[n][row_mask[n]]
        result.append(maps[n][r][:, cols[n]].mean(0))
    return torch.stack(result)


def test_readout_matches_eager_attention():
    torch.manual_seed(0)
    model = tiny_qwen2_5_vl(attn_implementation="sdpa").eval()
    input_ids = torch.randint(10, 100, (2, 20))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :5] = 0  # left padding in the second row
    rows = torch.tensor([[16, 17, 18], [17, 18, 19]])
    row_mask = torch.tensor([[True, True, False], [True, True, True]])
    cols = torch.tensor([[8, 9, 10, 11], [10, 11, 12, 13]])

    readout = AnswerReadout(rows, row_mask, cols)
    with torch.no_grad(), record_readout(model.model.language_model, readout):
        model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    got = readout.readout()
    with torch.no_grad():
        expected = _reference_readout(model, input_ids, attention_mask, rows, row_mask, cols)
    assert readout.num_layers == len(model.model.language_model.layers)
    assert torch.allclose(got, expected, atol=1e-5), (got, expected)


def test_readout_restores_attention_implementation():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model = tiny_qwen2_5_vl(attn_implementation="eager")
    lm = model.model.language_model
    sdpa_before = ALL_ATTENTION_FUNCTIONS["sdpa"]
    readout = AnswerReadout(torch.tensor([[3]]), torch.tensor([[True]]), torch.tensor([[1]]))
    with torch.no_grad(), record_readout(lm, readout):
        assert lm.layers[0].self_attn.config._attn_implementation == "sdpa"
        model(input_ids=torch.randint(10, 100, (1, 6)), return_dict=True)
    assert lm.layers[0].self_attn.config._attn_implementation == "eager"
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is sdpa_before
