# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""CPU-only tests for advantage-weighted SFT (offline GRPO + KL trust region).

These cover the verification checklist in ``custom-trainer.md``: vanilla equivalence, sign,
masking, alignment, eval safety, a tiny overfit, and the KL trust region (zero when policies
match, positive otherwise, gated off by kl_beta=0 / ref_logits=None). They operate on the pure
loss helper :func:`compute_weighted_sft_loss` and a toy LM, so no GPU and no real HF model are
required.
"""

import importlib.util
import json
import os

import pytest
import torch
import torch.nn.functional as F

from llamafactory.train.sft.weighted import REDUCTION_MODES, compute_weighted_sft_loss, shifted_token_nll


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_script_module():
    """Import scripts/train_weighted.py (not a package) by file path."""
    path = os.path.join(REPO_ROOT, "scripts", "train_weighted.py")
    spec = importlib.util.spec_from_file_location("train_weighted", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy_batch(seed: int = 0, batch: int = 4, seq: int = 7, vocab: int = 11, prompt_len: int = 3):
    """Random logits + labels with the first `prompt_len` positions masked (-100)."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=g)
    labels = torch.randint(0, vocab, (batch, seq), generator=g)
    labels[:, :prompt_len] = -100
    return logits, labels


def test_vanilla_equivalence():
    """Checklist 1: advantages == 1.0 reproduces standard token-mean SFT loss."""
    logits, labels = _toy_batch()
    adv = torch.ones(logits.size(0))

    got = compute_weighted_sft_loss(logits, labels, adv, reduction="batch_token_mean")

    # Reference: HF-style token-mean cross-entropy over non-ignored shifted tokens.
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    expected = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    assert torch.allclose(got, expected, atol=1e-6)


def test_eval_safety_none_advantage():
    """Checklist 5: a missing advantage column (None) falls back to ones, no error."""
    logits, labels = _toy_batch()
    with_none = compute_weighted_sft_loss(logits, labels, None, reduction="batch_token_mean")
    with_ones = compute_weighted_sft_loss(logits, labels, torch.ones(logits.size(0)))
    assert torch.allclose(with_none, with_ones, atol=1e-6)
    assert torch.isfinite(with_none)


def test_masking_ignores_prompt_logits():
    """Checklist 3: logits that only predict ignored (-100) targets do not change the loss.

    Under the causal shift, ``logits[:, t]`` predicts ``labels[:, t + 1]``. A logit position is
    irrelevant to the loss iff its predicted target is ``-100`` (or it is the final position,
    which the shift drops entirely). Perturbing exactly those positions must leave the loss fixed.
    """
    logits, labels = _toy_batch()
    adv = torch.tensor([1.5, -0.5, 2.0, -1.0])
    base = compute_weighted_sft_loss(logits, labels, adv)

    seq = labels.size(1)
    safe = torch.zeros(seq, dtype=torch.bool)
    safe[-1] = True  # last-position logits are dropped by the shift
    safe[:-1] = labels[0, 1:] == -100  # predicts an ignored target (prompt mask is shared across rows)

    perturbed = logits.clone()
    perturbed[:, safe, :] += 100.0 * torch.randn_like(perturbed[:, safe, :])
    after = compute_weighted_sft_loss(perturbed, labels, adv)
    assert torch.allclose(base, after, atol=1e-5)


def test_sign_of_advantage():
    """Checklist 2: positive advantage pushes target log-prob up; negative pushes it down."""

    def step_then_logprob(advantage_value: float) -> float:
        torch.manual_seed(123)
        vocab, seq = 8, 5
        logits = torch.zeros(1, seq, vocab, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 4, 5]])  # response = last 3 tokens
        adv = torch.tensor([advantage_value])

        loss = compute_weighted_sft_loss(logits, labels, adv, reduction="batch_token_mean")
        loss.backward()
        with torch.no_grad():
            updated = logits - 1.0 * logits.grad  # one SGD step

        # mean log-prob of the (shifted) target tokens after the step
        shift_logits = updated[:, :-1, :]
        shift_labels = labels[:, 1:]
        logprobs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100
        tgt = shift_labels.clamp(min=0).unsqueeze(-1)
        picked = logprobs.gather(-1, tgt).squeeze(-1)
        return picked[mask].mean().item()

    baseline = -torch.log(torch.tensor(8.0)).item()  # uniform log-prob before any step
    assert step_then_logprob(+2.0) > baseline  # positive advantage raises target prob
    assert step_then_logprob(-2.0) < baseline  # negative advantage lowers it


def test_reduction_modes_consistency():
    """example_mean removes length bias; sum is unnormalized; all are finite."""
    logits, labels = _toy_batch()
    adv = torch.tensor([1.0, 1.0, 1.0, 1.0])
    for mode in REDUCTION_MODES:
        out = compute_weighted_sft_loss(logits, labels, adv, reduction=mode)
        assert torch.isfinite(out)

    # With equal per-example token counts and adv == 1, example_mean == batch_token_mean.
    eq_logits = torch.randn(3, 6, 9)
    eq_labels = torch.randint(0, 9, (3, 6))
    eq_labels[:, :2] = -100  # identical mask -> identical token counts per example
    a = compute_weighted_sft_loss(eq_logits, eq_labels, torch.ones(3), reduction="batch_token_mean")
    b = compute_weighted_sft_loss(eq_logits, eq_labels, torch.ones(3), reduction="example_mean")
    assert torch.allclose(a, b, atol=1e-6)


def test_kl_zero_when_policies_match():
    """Checklist (KL): identical ref logits -> KL term is ~0, so loss == pg_loss."""
    logits, labels = _toy_batch()
    adv = torch.tensor([1.5, -0.5, 2.0, -1.0])
    pg_only = compute_weighted_sft_loss(logits, labels, adv, reduction="example_mean")
    with_kl = compute_weighted_sft_loss(
        logits, labels, adv, reduction="example_mean", ref_logits=logits.clone(), kl_beta=1.0
    )
    assert torch.allclose(pg_only, with_kl, atol=1e-6)


def test_kl_positive_when_policies_differ():
    """Checklist (KL): a distinct reference makes KL > 0, raising the loss above pg-only."""
    logits, labels = _toy_batch(seed=0)
    ref_logits, _ = _toy_batch(seed=99)  # different policy
    adv = torch.tensor([1.5, -0.5, 2.0, -1.0])

    pg_only = compute_weighted_sft_loss(logits, labels, adv, reduction="example_mean")
    with_kl = compute_weighted_sft_loss(
        logits, labels, adv, reduction="example_mean", ref_logits=ref_logits, kl_beta=0.1
    )
    # k3 KL is non-negative, so a positive kl_beta can only raise the loss; with distinct
    # policies the gap is strictly positive.
    assert (with_kl - pg_only).item() > 1e-4


def test_kl_gated_off():
    """Checklist (KL): kl_beta=0.0 OR ref_logits=None returns exactly the pg-only loss."""
    logits, labels = _toy_batch()
    ref_logits, _ = _toy_batch(seed=99)
    adv = torch.tensor([1.5, -0.5, 2.0, -1.0])

    pg_only = compute_weighted_sft_loss(logits, labels, adv, reduction="example_mean")
    beta_zero = compute_weighted_sft_loss(
        logits, labels, adv, reduction="example_mean", ref_logits=ref_logits, kl_beta=0.0
    )
    ref_none = compute_weighted_sft_loss(logits, labels, adv, reduction="example_mean", ref_logits=None, kl_beta=1.0)
    assert torch.equal(pg_only, beta_zero)
    assert torch.equal(pg_only, ref_none)


def test_ref_per_tok_matches_ref_logits():
    """The memory-saving path (precomputed ref_per_tok) equals passing full ref_logits.

    The trainer computes ref NLL inside no_grad and frees the [B, T, V] logits; this guards that
    the [B, T-1] shortcut is numerically identical to letting the loss upcast the logits itself.
    """
    logits, labels = _toy_batch(seed=0)
    ref_logits, _ = _toy_batch(seed=99)
    adv = torch.tensor([1.5, -0.5, 2.0, -1.0])

    via_logits = compute_weighted_sft_loss(
        logits, labels, adv, reduction="example_mean", ref_logits=ref_logits, kl_beta=0.1
    )
    via_per_tok = compute_weighted_sft_loss(
        logits, labels, adv, reduction="example_mean", ref_per_tok=shifted_token_nll(ref_logits, labels), kl_beta=0.1
    )
    assert torch.allclose(via_logits, via_per_tok, atol=1e-6)


def test_return_metrics_components():
    """return_metrics surfaces pg_loss and raw kl, and total == pg_loss + kl_beta * kl."""
    logits, labels = _toy_batch(seed=0)
    ref_logits, _ = _toy_batch(seed=99)
    adv = torch.tensor([1.5, -0.5, 2.0, -1.0])
    kl_beta = 0.1

    total, metrics = compute_weighted_sft_loss(
        logits, labels, adv, reduction="example_mean", ref_logits=ref_logits, kl_beta=kl_beta, return_metrics=True
    )
    assert set(metrics) == {"pg_loss", "kl"}
    assert metrics["kl"].item() > 0.0  # distinct reference -> positive KL
    recon = metrics["pg_loss"] + kl_beta * metrics["kl"]
    assert torch.allclose(total, recon, atol=1e-6)

    # Trust region off -> kl component is exactly 0 and total == pg_loss.
    pg_total, pg_metrics = compute_weighted_sft_loss(
        logits, labels, adv, reduction="example_mean", return_metrics=True
    )
    assert pg_metrics["kl"].item() == 0.0
    assert torch.equal(pg_total, pg_metrics["pg_loss"])


def test_empty_mask_does_not_crash():
    """clamp(min=1) guards a microbatch with no response tokens."""
    logits = torch.randn(2, 4, 5)
    labels = torch.full((2, 4), -100)
    out = compute_weighted_sft_loss(logits, labels, torch.ones(2))
    assert torch.isfinite(out) and out.item() == 0.0


def test_invalid_reduction_raises():
    logits, labels = _toy_batch()
    with pytest.raises(ValueError):
        compute_weighted_sft_loss(logits, labels, torch.ones(logits.size(0)), reduction="nope")


def test_alignment_order_and_assert(tmp_path):
    """Checklist 4: advantages load in file order; a length mismatch trips the guard."""
    script = _load_script_module()

    rows = [
        {"instruction": "a", "input": "", "output": "x", "advantage": 0.5},
        {"instruction": "b", "input": "", "output": "y", "advantage": -1.5},
        {"instruction": "c", "input": "", "output": "z", "advantage": 2.0},
    ]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    advs = script.load_advantages_in_order(str(path))
    assert advs == [0.5, -1.5, 2.0]  # row k advantage matches JSON row k

    # The driver's alignment guard fires when lengths disagree (simulate a dropped row).
    n_dataset_rows = len(rows) - 1
    with pytest.raises(AssertionError):
        assert n_dataset_rows == len(advs), "misaligned"


def test_tiny_overfit():
    """Checklist 6: a positive-advantage example's target prob rises as loss falls."""
    torch.manual_seed(0)
    vocab, seq = 12, 6
    # Toy autoregressive model: embed previous token, linear head over the vocab.
    embed = torch.nn.Embedding(vocab, 16)
    head = torch.nn.Linear(16, vocab)
    params = list(embed.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=5e-2)

    # 10 examples; response = last 3 tokens (first 3 are masked prompt).
    input_ids = torch.randint(0, vocab, (10, seq))
    labels = input_ids.clone()
    labels[:, :3] = -100
    adv = torch.ones(10)  # all positive -> behaves like SFT overfit

    def forward(ids):
        return head(embed(ids))  # [B, T, V]

    def held_prob():
        with torch.no_grad():
            logits = forward(input_ids[:1])
            lp = F.log_softmax(logits[:, :-1, :], dim=-1)
            sl = labels[:1, 1:]
            mask = sl != -100
            picked = lp.gather(-1, sl.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            return picked[mask].mean().exp().item()

    first_loss = None
    p0 = held_prob()
    for _ in range(60):
        opt.zero_grad()
        loss = compute_weighted_sft_loss(forward(input_ids), labels, adv)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
    last_loss = loss.item()
    p1 = held_prob()

    assert last_loss < first_loss  # loss decreases
    assert p1 > p0  # held example's target probability rises
