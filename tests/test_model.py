import torch
from transformers import GenerationConfig

from realvr.model.qwen_lvr import QwenWithLVR
from tests.tiny_model import IMAGE_TOKEN_ID, LVR_END_ID, LVR_START_ID, VISION_END_ID, VISION_START_ID, tiny_qwen2_5_vl

K = 3


def _image_inputs(batch_size=2, grid=(1, 4, 4)):
    """Random pixels for one 4x4-patch image per row (-> 2x2 = 4 merged visual tokens)."""
    t, h, w = grid
    patches = t * h * w
    pixel_values = torch.randn(batch_size * patches, 3 * 2 * 14 * 14)
    image_grid_thw = torch.tensor([list(grid)] * batch_size)
    return pixel_values, image_grid_thw, patches // 4


def _prompt(batch_size=2, n_image_tokens=4, pad_first=2):
    """Prompts with one image each; the first row is shorter and left-padded by `pad_first` tokens."""
    image = [VISION_START_ID] + [IMAGE_TOKEN_ID] * n_image_tokens + [VISION_END_ID]
    rows = [[11, 21, 22] + image + [12, 13, 14] for _ in range(batch_size)]
    rows[0] = [0] * pad_first + rows[0][pad_first:]
    ids = torch.tensor(rows)
    mask = (torch.arange(ids.shape[1])[None, :] >= torch.tensor([pad_first] + [0] * (batch_size - 1))[:, None]).long()
    return ids, mask


def _model():
    model = tiny_qwen2_5_vl(model_class=QwenWithLVR).eval()
    for p in model.visual.parameters():
        p.requires_grad_(False)
    return model


def test_generation_records_the_fed_latent_states():
    model = _model()
    ids, mask = _prompt()
    pixel_values, grid, _ = _image_inputs()
    config = GenerationConfig(max_new_tokens=K + 3, do_sample=False, pad_token_id=0, eos_token_id=None,
                              return_dict_in_generate=True, use_cache=True)
    out = model.generate(input_ids=ids, attention_mask=mask, pixel_values=pixel_values, image_grid_thw=grid,
                         generation_config=config, lvr_steps=K, force_lvr_start=True)
    seq = out.sequences
    P = ids.shape[1]
    assert (seq[:, P] == LVR_START_ID).all()
    assert out.latent_mask.all()

    # Manual recurrence with full recomputation (no cache): z_1 = h(<lvr_start>), z_t = h(z_{t-1}).
    with torch.no_grad():
        cur = torch.cat([ids, seq[:, P:P + 1]], dim=1)
        cur_mask = torch.cat([mask, torch.ones_like(seq[:, P:P + 1])], dim=1)
        states = []
        for t in range(K):
            L = cur.shape[1]
            lvr_mask = torch.zeros(cur.shape, dtype=torch.bool)
            lvr_states = torch.zeros(*cur.shape, model.config.text_config.hidden_size)
            for j, z in enumerate(states):
                lvr_mask[:, P + 1 + j] = True
                lvr_states[:, P + 1 + j] = z
            o = model(input_ids=cur, attention_mask=cur_mask, pixel_values=pixel_values, image_grid_thw=grid,
                      lvr_states=lvr_states, lvr_mask=lvr_mask)
            states.append(o.last_position_hidden_state)
            cur = torch.cat([cur, seq[:, L:L + 1]], dim=1)
            cur_mask = torch.cat([cur_mask, torch.ones_like(seq[:, L:L + 1])], dim=1)
    manual = torch.stack(states, dim=1)
    assert torch.allclose(out.latent_states, manual, atol=1e-4), (out.latent_states - manual).abs().max()


def test_left_padded_rows_generate_as_if_alone():
    model = _model()
    ids, mask = _prompt(pad_first=2)
    pixel_values, grid, _ = _image_inputs()
    config = GenerationConfig(max_new_tokens=K + 3, do_sample=False, pad_token_id=0, eos_token_id=None,
                              return_dict_in_generate=True, use_cache=True)
    batched = model.generate(input_ids=ids, attention_mask=mask, pixel_values=pixel_values, image_grid_thw=grid,
                             generation_config=config, lvr_steps=K, force_lvr_start=True)
    alone = model.generate(input_ids=ids[:1, 2:], attention_mask=mask[:1, 2:], pixel_values=pixel_values[:16],
                           image_grid_thw=grid[:1], generation_config=config, lvr_steps=K, force_lvr_start=True)
    assert torch.allclose(batched.latent_states[0], alone.latent_states[0], atol=1e-4)
    assert torch.equal(batched.sequences[0, 2:], alone.sequences[0])


def test_teacher_forcing_reproduces_generation_logits():
    model = _model()
    ids, mask = _prompt()
    pixel_values, grid, _ = _image_inputs()
    config = GenerationConfig(max_new_tokens=K + 4, do_sample=False, pad_token_id=0, eos_token_id=None,
                              return_dict_in_generate=True, output_scores=True, use_cache=True)
    out = model.generate(input_ids=ids, attention_mask=mask, pixel_values=pixel_values, image_grid_thw=grid,
                         generation_config=config, lvr_steps=K, force_lvr_start=True)
    seq = out.sequences
    P = ids.shape[1]
    full_mask = torch.cat([mask, torch.ones_like(seq[:, P:])], dim=1)
    lvr_mask = torch.zeros(seq.shape, dtype=torch.bool)
    lvr_mask[:, P + 1:P + 1 + K] = True
    lvr_states = torch.zeros(*seq.shape, model.config.text_config.hidden_size)
    lvr_states[:, P + 1:P + 1 + K] = out.latent_states
    with torch.no_grad():
        logits = model(input_ids=seq, attention_mask=full_mask, pixel_values=pixel_values, image_grid_thw=grid,
                       lvr_states=lvr_states, lvr_mask=lvr_mask).logits
    # greedy tokens after the latent span must be the argmax of the teacher-forced logits
    for pos in range(P + K + 1, seq.shape[1]):
        assert torch.equal(logits[:, pos - 1].argmax(-1), seq[:, pos])


def test_only_the_first_lvr_start_opens_a_latent_span():
    model = _model()
    ids, mask = _prompt(batch_size=1, pad_first=0)
    ids = torch.cat([ids, torch.tensor([[LVR_START_ID, 30, 31, 32, LVR_END_ID, LVR_START_ID]])], dim=1)
    mask = torch.ones_like(ids)
    pixel_values, grid, _ = _image_inputs(batch_size=1)
    config = GenerationConfig(max_new_tokens=K + 2, do_sample=False, pad_token_id=0, eos_token_id=None,
                              return_dict_in_generate=True, use_cache=True)
    out = model.generate(input_ids=ids, attention_mask=mask, pixel_values=pixel_values, image_grid_thw=grid,
                         generation_config=config, lvr_steps=K)
    # the prompt already ends with <|lvr_start|>, so one span of K latent steps follows
    assert int(out.latent_mask.sum()) == K
