"""CPU-only checks of the evil-Michael merge helpers on tiny synthetic tensors."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_split_and_unembed_delta_reproduce_the_hand_computed_logits():
    from hl.teachers import merge_evil_michael as m

    torch.manual_seed(0)
    r, hidden, vocab = 2, 8, 12
    cfg = {"r": r, "lora_alpha": 8, "use_rslora": False, "modules_to_save": None}
    P = m.UNEMBED_PREFIX
    sd = {
        P + "lora_A.weight": torch.randn(r, hidden),
        P + "lora_B.weight": torch.randn(vocab, r),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(r, hidden),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(hidden, r),
    }
    unembed, standard, scale = m.split_adapter_state(sd, cfg, expect_counts=(2, 2))
    assert scale == 4.0 and set(unembed) == {P + "lora_A.weight", P + "lora_B.weight"}
    assert all("unembed_tokens" not in k for k in standard)  # what the stock adapter loader is given

    head = torch.randn(vocab, hidden)
    delta = m.unembed_delta(unembed, cfg, scale, head)
    h = torch.randn(3, hidden)
    logits_merged = h @ (head + delta).T
    logits_by_hand = h @ head.T + scale * (h @ sd[P + "lora_A.weight"].T) @ sd[P + "lora_B.weight"].T
    assert torch.allclose(logits_merged, logits_by_hand, atol=1e-4)
    # the trap this script exists for: dropping the two tensors changes the logits
    assert not torch.allclose(logits_merged, h @ head.T, atol=1e-2)


def test_split_rejects_unexpected_adapter_shapes():
    from hl.teachers import merge_evil_michael as m

    with pytest.raises(AssertionError):
        m.split_adapter_state({"a": torch.zeros(1)}, {"r": 2, "lora_alpha": 8})  # not 2 + 504 tensors
    with pytest.raises(AssertionError):
        m.split_adapter_state({}, {"r": 2, "lora_alpha": 8, "use_rslora": True}, expect_counts=(0, 0))
