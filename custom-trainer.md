# Handoff: Advantage-Weighted SFT Trainer for LlamaFactory (offline GRPO + KL trust region)

## Context

We run **expert iteration** for multi-turn Lean theorem proving. Rollouts are generated
externally (vLLM serving Qwen3 + a Lean environment), **not** by any RL framework. Each
*session* is a sequence of up to 8 proof edits; we run 16 sessions per theorem over ~2500
theorems per iteration.

For training, **each edit is an independent single-turn SFT example**: prompt = (system +
proof state), response = the edit. There is no multi-turn `history`. Sessions are flattened
into per-edit examples.

We are replacing plain filtered-SFT with **advantage-weighted SFT** = offline GRPO. Each
example carries a scalar `advantage = reward - baseline`, where the baseline is the
per-theorem success rate (fraction of that theorem's 16 sessions that succeeded). Positive
advantage (edits from successful sessions) raises the edit's probability; negative advantage
(failed sessions) lowers it.

We run our **own entry point script**, not `llamafactory-cli`.

## IMPORTANT: why a KL trust region is REQUIRED

A first run *without* a trust region diverged. Loss started small and negative as expected,
then once LR reached peak it ran away: −0.07 → −0.28 → −0.74 → −3.2 → −7.8 → −15, with
grad_norm climbing 0.15 → 46. This is the **unbounded-negative-gradient collapse**: for a
negative-advantage edit the loss term is `adv * nll` with `adv < 0`, so the optimizer
minimizes it by driving `nll → ∞` (policy probability on those tokens → 0), with no lower
bound. The ~25% positive-advantage edits cannot counterbalance.

The fix is a **KL penalty to a reference policy** (the trust region GRPO/PPO always include).
It does NOT require sampling-time / vLLM logprobs (that's for PPO importance-ratio clipping).
It only needs the reference model's logprobs at train time, which for **LoRA is cheap**: the
reference is the same model with the adapter disabled. `kl_beta = 0.0` reproduces the
unstable behavior and must not be used with negative advantages.

## Objective the loss implements

    L = pg_term + kl_beta * KL(policy || ref)
    pg_term ∝ Σ_examples A_i * Σ_tokens NLL(token)

Minimizing `A * NLL` raises log-prob of positive-advantage edits and lowers it for negative
ones; the KL term bounds how far the policy may drift from the reference, preventing collapse.

## Files (ALL ADDITIVE — do not edit any upstream file)

1. `src/llamafactory/train/sft/weighted.py` — `compute_weighted_sft_loss`, `WeightedSFTTrainer`,
   `WeightedCollator`.
2. `scripts/train_weighted.py` — `run_weighted_sft` + `__main__`.
3. `tests/train/test_weighted_sft.py` — CPU-only verification tests.

Do NOT modify `tuner.py`, `cli.py`, the hparams/stage enum, or any `dataset_info` column
mapping. Keeping changes additive keeps the fork rebaseable. None of the new data fields
(`theorem_id`, `session_id`, `reward`, `advantage`) go through LlamaFactory's column system —
`advantage` is spliced in via `add_column` after tokenization; the rest are consumed upstream
in data-prep and ignored by LF's loader.

## As implemented (deviations from a bare reference)

The reduction mode and KL coefficient are **constructor kwargs** on `WeightedSFTTrainer`
(`weighted_reduction`, `kl_beta`), not hard-coded class attributes. The entry-point script
reads optional top-level YAML keys `weighted_reduction` (default `example_mean`) and `kl_beta`
(default `0.05`), strips them before LlamaFactory's parser sees the config, and passes them to
the trainer. Defaults: `reduction = example_mean`, `kl_beta = 0.05`.

The reference pass is guarded: if `kl_beta > 0` but the unwrapped model has no
`disable_adapter()` (i.e. not a PEFT/LoRA model), the trainer raises a clear error rather than
silently running without a trust region.

## Loss helper signature

```python
def compute_weighted_sft_loss(
    logits, labels, advantages=None,
    reduction="example_mean", ref_logits=None, kl_beta=0.0,
) -> torch.Tensor
```

- fp32-upcast causal-shifted cross-entropy, `reduction="none"`, gives `per_tok = -logp_policy`.
- `mask = (shift_labels != -100)` selects response (edit) tokens only.
- `advantages is None` → ones (eval-safe vanilla SFT); else moved to the logits' device/dtype.
- `weighted = per_tok * mask * advantages[:, None]`, reduced per `reduction`.
- KL is added **only** when `ref_logits is not None and kl_beta != 0.0`:
  - `ref_per_tok = -logp_ref` (same fp32-shifted CE on `ref_logits`).
  - `log_ratio = per_tok - ref_per_tok == logp_ref - logp_policy`.
  - `kl_per_tok = exp(log_ratio) - log_ratio - 1` — the **k3 estimator** of `KL(policy || ref)`,
    always ≥ 0.
  - `kl = (kl_per_tok * mask).sum() / mask.sum().clamp(min=1)` (batch-token-mean for the KL).
  - returns `pg_loss + kl_beta * kl`.

## Reference pass (in `WeightedSFTTrainer.compute_loss`)

```python
advantages = inputs.pop("advantage", None)
outputs = model(**inputs)

ref_logits = None
if self.kl_beta > 0.0 and advantages is not None:        # no ref / no KL during eval
    unwrapped = self.accelerator.unwrap_model(model)
    if not hasattr(unwrapped, "disable_adapter"):
        raise RuntimeError(...)                          # LoRA required for the reference
    with torch.no_grad(), unwrapped.disable_adapter():   # reference = base (LoRA off)
        ref_logits = model(**inputs).logits
        ref_per_tok = shifted_token_nll(ref_logits, inputs["labels"])  # [B, T-1]
        del ref_logits                                   # free the [B, T, V] tensor now

loss = compute_weighted_sft_loss(
    outputs.logits, inputs["labels"], advantages,
    reduction=self.weighted_reduction, ref_per_tok=ref_per_tok, kl_beta=self.kl_beta,
)
```

Notes on the reference pass:
- `disable_adapter()` is a PEFT/LoRA context manager. We `unwrap_model` to reach it, set the
  adapter-disabled flag on the underlying LoRA layers, then run the forward through the
  (still-wrapped) `model` so the flag is respected under the distributed wrapper. Confirm this
  behaves under your distributed setup (single-node multi-GPU DDP).
- The ref forward is `no_grad`, so it adds ~50% forward compute and no backward. Acceptable.
- **Memory**: only the reference *per-token NLL* (`[B, T-1]`) is needed, not the full `[B, T, V]`
  logits. We collapse the logits inside the `no_grad` block via `shifted_token_nll` and `del` them
  immediately, so the vocab-sized reference tensor (Qwen3 vocab ~152k → multi-GB at long context)
  never coexists with the policy backward graph. Keeping the full `ref_logits` alive until loss
  time roughly doubles peak logit memory and is a likely OOM cause; this path avoids it. The loss
  helper still accepts `ref_logits` directly (used by the unit tests) as an equivalent fallback.
- **Reference modes** (`kl_reference` in the YAML, wired via `kl_ref_adapter_name`):
  - `base` — disable the adapter; reference = bare base model. Correct only when the starting
    adapter is fresh/zero or has been merged into the base you loaded.
  - `start_adapter` — load the resumed adapter a second time, frozen, as a `kl_reference`
    adapter; reference = base + start-of-iteration adapter = the data-generating (rollout)
    policy. This is the correct GRPO reference when **continuing an adapter across iterations
    without merging** (our case). The ref pass activates the frozen adapter via `set_adapter`
    and restores the trainable `default` adapter in a `finally` before backward (mirrors TRL's
    `ref_adapter_name`). Costs one extra (small) frozen adapter in memory, not a second base.
  - Default: `start_adapter` is auto-selected when an adapter is being resumed
    (`adapter_name_or_path` set and not `create_new_adapter`) and `kl_beta > 0`; else `base`.

## Running it

    python scripts/train_weighted.py train.yaml data/your_dataset.json

`train.yaml` uses the same keys as a LlamaFactory SFT config, plus optional `weighted_reduction`
and `kl_beta`. `data/your_dataset.json` is the file referenced by your `dataset_info` entry;
advantages are read from it in file order and `add_column`'d onto the tokenized train dataset.

## Reduction (decided: per-example mean)

`weighted_reduction = "example_mean"` is the intended default. Edit verbosity varies a lot here
(one-line closers like `omega` vs. long structured blocks) and the advantage is assigned per
edit, so each edit should contribute its advantage once regardless of length. `batch_token_mean`
would let verbose edits dominate by length. Keep `batch_token_mean` available — verification
test 1 (vanilla equivalence) depends on it. Do not delete it. `sum` is unnormalized (Dr. GRPO
style); rely on the learning rate.

## Trust-region / stability hyperparameters

- `kl_beta = 0.05` to start. Raise toward 0.1–0.2 if the loss still drifts away / grad_norm
  keeps climbing; lower it if training stalls (loss and grad_norm pinned near 0 = over-constrained).
- `max_grad_norm = 1.0` — ALWAYS on. The diverged run had no gradient clipping (grad_norm hit 46).
- Learning rate ~1e-5 to start. The diverged run used peak 1e-4, which was too hot.
- Watch grad_norm in wandb: it should stay roughly O(0.1–1). A sustained climb past a few is the
  early warning of divergence — raise `kl_beta` and/or lower LR.
- With the KL term, expect small, bounded loss values that do NOT run away. A monotonic dive into
  large-magnitude negatives means the trust region is too weak (raise `kl_beta`).

## Data assumptions and advantage computation (upstream data-prep)

Each JSON row has the standard alpaca fields (`instruction`, `input`, `output`, `system`) plus
ignored extras `theorem_id`, `session_id`, `reward`, `advantage`.

Compute the baseline over **sessions, not edits** (dedup by `session_id` first — averaging over
per-edit rows would give a length-weighted baseline, which is wrong). Then apply **per-session
edit weighting**: divide each edit's advantage by the number of edits in its session, so every
session contributes equally. This corrects the over-representation of long (failed) sessions —
failures run the full 8 edits while successes terminate early, which otherwise biases the
edit-level advantage net-negative and feeds the collapse.

```python
from collections import defaultdict, Counter
from statistics import mean

edits_per_session = Counter(ex.session_id for ex in examples)

sessions_by_theorem = defaultdict(dict)              # theorem_id -> {session_id: reward}
for ex in examples:
    sessions_by_theorem[ex.theorem_id][ex.session_id] = ex.reward
baseline = {tid: mean(s.values()) for tid, s in sessions_by_theorem.items()}

for ex in examples:
    raw_adv = ex.reward - baseline[ex.theorem_id]    # GRPO group-relative advantage
    ex.advantage = raw_adv / edits_per_session[ex.session_id]   # per-session weighting

advantages_in_order = [ex.advantage for ex in examples]   # same order as the JSON
```

Note: the `1/n_edits` division shrinks advantage magnitudes (~8x), which interacts with `kl_beta`
and LR — tune together. Optionally drop degenerate groups (baseline 0 or 1 -> all advantages 0)
here; at 0.2–0.3 success that's only ~1–3% of theorems.

## Invariants & edge cases

- **Eval safety**: advantage is attached only to the train dataset. Both the collator (default
  1.0) and `compute_weighted_sft_loss` (`advantages=None` -> ones) must not crash on eval. The
  trainer also skips the reference pass / KL when advantages are absent.
- **Packing off**: assert `data_args.packing` and `neat_packing` are false (guards included).
- **Alignment**: build `advantages_in_order` from the same JSON LF loads; keep the length assert;
  keep `cutoff_len` generous so no row is length-filtered.
- **fp32 CE** for both policy and reference logits.
- **Device/dtype**: advantages moved to the logits' device/dtype.
- **Gradient accumulation**: the custom reductions normalize per-microbatch, so with
  `grad_accum > 1` cross-microbatch scaling is approximate. Start at `grad_accum = 1`.
- **Auxiliary losses**: recomputing CE from logits drops `outputs.loss` and any aux loss. Fine for
  dense Qwen3; a Qwen3-MoE variant would lose the router aux loss — add it back from `outputs`.

## Verification checklist (implemented as tests in `tests/train/test_weighted_sft.py`)

1. **Vanilla equivalence (anchor)** — `reduction="batch_token_mean"`, `advantages=ones`,
   `ref_logits=None`: loss equals HF's standard token-mean SFT loss within fp tolerance.
   Confirms shift + mask. Does NOT hold for `example_mean`.
2. **KL is zero when policies match** — `ref_logits = logits.clone()`, `kl_beta = 1.0`: the KL
   term is ~0, so loss == pg_loss within fp tolerance.
3. **KL is positive otherwise** — distinct `ref_logits`, `kl_beta > 0`: kl > 0 and the returned
   loss exceeds the pg-only loss.
4. **kl_beta / ref gating** — `kl_beta = 0.0` or `ref_logits = None` returns exactly `pg_loss`.
5. **Sign** — one step with a positive advantage raises mean log-prob of target tokens; negative
   advantage lowers it (kl off, tiny batch).
6. **Masking** — perturbing prompt-position logits (label -100) does not change the loss.
7. **Alignment** — advantages load in file order; the length assert fires on a truncated list.
8. **Eval safety** — `advantages=None` falls back to ones without error and takes no ref pass.
9. **Reduction/edge** — modes finite; `example_mean == batch_token_mean` under equal token
   counts; empty mask returns 0 (no crash); invalid reduction raises.

Not in the CPU test file: the end-to-end **stability smoke test** (a short real run with
`kl_beta = 0.05`, `max_grad_norm = 1.0`, lr 1e-5 must keep loss bounded and grad_norm O(1)) —
run that as a GPU regression for the collapse.

## Confirm against the installed fork (versions drift)

- `CustomSeq2SeqTrainer.compute_loss` is `(self, model, inputs, *args, **kwargs)` — our override
  `(self, model, inputs, return_outputs=False, **kwargs)` absorbs `num_items_in_batch` safely.
- That `model.disable_adapter()` works through your distributed wrapper after `unwrap_model`
  (single-node multi-GPU DDP).
- Collator class name + kwargs in `llamafactory/data/collator.py`
  (`SFTDataCollatorWith4DAttentionMask`).
- `get_dataset` signature and `dataset_module` keys.
- That `CustomSeq2SeqTrainer` lives at `llamafactory/train/sft/trainer.py`.
