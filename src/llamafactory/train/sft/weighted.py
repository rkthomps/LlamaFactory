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

r"""Advantage-weighted SFT (offline GRPO) trainer and collator, with a KL trust region.

This is an additive fork file: it does not touch any upstream module. Each training
example carries a scalar ``advantage = reward - baseline`` (see ``custom-trainer.md``).
The loss is a surrogate whose gradient is the one-step policy-gradient / GRPO estimator,
plus a KL penalty to a reference policy::

    L  =  pg_term  +  kl_beta · KL(policy || ref)
    pg_term  ∝  Σ_examples  A_i · Σ_tokens NLL(token_t | context)

We recompute the per-token, unreduced cross-entropy from the logits so we can multiply by
the per-example advantage *before* reducing. The model's built-in ``outputs.loss`` cannot be
used because it is already reduced to a single unweighted scalar.

The KL term is REQUIRED for stability. Without it, a negative-advantage edit contributes
``adv · nll`` with ``adv < 0``, which the optimizer minimizes by driving ``nll -> inf``
(policy probability on those tokens -> 0) with no lower bound — the unbounded-negative-gradient
collapse (loss running away to large negatives, grad_norm climbing). The KL penalty to the
reference (for LoRA: the same model with the adapter disabled) bounds how far the policy may
drift and prevents the collapse. ``kl_beta = 0.0`` reproduces the unstable behavior and must
not be used with negative advantages.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch

from ...data.collator import SFTDataCollatorWith4DAttentionMask
from .trainer import CustomSeq2SeqTrainer


# Reduction over the per-token weighted NLL. See "Reduction decision" in custom-trainer.md.
REDUCTION_MODES = ("batch_token_mean", "example_mean", "sum")


def shifted_token_nll(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Per-token NLL ``[B, T-1]`` over causal-shifted, fp32-upcast logits (``-100`` -> 0 weight).

    Factored out so the reference policy's NLL can be computed *inside* the trainer's ``no_grad``
    block and the ``[B, T, V]`` reference logits freed before backward — see
    :meth:`WeightedSFTTrainer.compute_loss`. Keeping the full vocab-sized reference logits alive
    until loss time is what makes the KL pass memory-hungry.
    """
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.size())  # [B, T-1]  ==  -logp(token)


def compute_weighted_sft_loss(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    advantages: Optional["torch.Tensor"] = None,
    reduction: str = "example_mean",
    ref_logits: Optional["torch.Tensor"] = None,
    ref_per_tok: Optional["torch.Tensor"] = None,
    kl_beta: float = 0.0,
    return_metrics: bool = False,
):
    r"""Advantage-weighted SFT loss + optional KL-to-reference trust region (offline GRPO).

    Args:
        logits: ``[B, T, V]`` policy logits (any float dtype; upcast to fp32 internally).
        labels: ``[B, T]`` labels with ``-100`` on non-response (prompt / pad) positions.
        advantages: ``[B]`` per-example advantages. ``None`` falls back to ones, i.e. vanilla
            SFT — this is the eval-safe path when no advantage column is attached.
        reduction: one of :data:`REDUCTION_MODES`.

            - ``example_mean`` (default): per-example token-mean, then mean over examples. Each
              edit contributes its advantage equally regardless of token count (no length bias).
              This is the intended default — edit verbosity varies a lot (a one-line ``omega``
              vs. a long structured block) and the advantage is assigned per edit.
            - ``batch_token_mean``: ``Σ(A·NLL·mask) / Σ mask``. Scale-stable; longer edits exert
              more total pull (length-weighted). Kept available — verification test 1 (vanilla
              equivalence) uses it. Do not delete it.
            - ``sum``: pure sum, no normalization (Dr. GRPO style); rely on the learning rate.
        ref_logits: ``[B, T, V]`` reference-policy logits (e.g. the LoRA adapter disabled). When
            provided with ``kl_beta > 0``, adds ``kl_beta · KL(policy || ref)`` over the response
            tokens using the k3 estimator. REQUIRED for stability with negative advantages.
        ref_per_tok: ``[B, T-1]`` precomputed reference NLL (from :func:`shifted_token_nll`), an
            alternative to ``ref_logits`` that avoids materializing the full vocab-sized reference
            logits at loss time — the trainer uses this path for memory. Takes precedence over
            ``ref_logits`` when both are given.
        kl_beta: KL coefficient. ``0.0`` disables the trust region (UNSTABLE with negatives — see
            the module docstring). The KL term is also skipped entirely when neither ``ref_logits``
            nor ``ref_per_tok`` is given (e.g. during eval, where no reference pass is run).
        return_metrics: if ``True``, also return a dict of the detached scalar components
            ``{"pg_loss", "kl"}`` for logging (``kl`` is the raw mean per-token KL, before
            multiplication by ``kl_beta``). ``kl`` is 0 when the trust region is disabled.

    Returns:
        The scalar loss tensor, or ``(loss, metrics)`` when ``return_metrics`` is set.
    """
    if reduction not in REDUCTION_MODES:
        raise ValueError(f"Unknown reduction {reduction!r}; expected one of {REDUCTION_MODES}.")

    # causal shift + fp32 upcast for numerically stable CE under bf16/fp16
    shift_labels = labels[..., 1:].contiguous()
    per_tok = shifted_token_nll(logits, labels)  # [B, T-1]  ==  -logp_policy(token)

    mask = (shift_labels != -100).to(per_tok.dtype)  # response (edit) tokens only

    if advantages is None:  # eval / no advantage column: behave like vanilla SFT
        advantages = torch.ones(per_tok.size(0), device=per_tok.device, dtype=per_tok.dtype)
    else:
        advantages = advantages.to(device=per_tok.device, dtype=per_tok.dtype)

    weighted = per_tok * mask * advantages.unsqueeze(1)  # [B, T]

    if reduction == "batch_token_mean":
        # clamp(min=1) guards a microbatch with no response tokens (empty mask)
        pg_loss = weighted.sum() / mask.sum().clamp(min=1)
    elif reduction == "example_mean":
        tokens_per_example = mask.sum(dim=1).clamp(min=1)
        pg_loss = (weighted.sum(dim=1) / tokens_per_example).mean()
    else:  # "sum"
        pg_loss = weighted.sum()

    if kl_beta == 0.0 or (ref_logits is None and ref_per_tok is None):
        if return_metrics:
            return pg_loss, {"pg_loss": pg_loss.detach(), "kl": pg_loss.detach() * 0.0}
        return pg_loss

    # KL(policy || ref) over response tokens, k3 estimator (always >= 0). This is the trust
    # region that bounds policy drift and prevents the negative-advantage collapse. Prefer the
    # precomputed ref_per_tok (cheap [B, T-1]); only upcast full ref logits as a fallback.
    if ref_per_tok is None:
        ref_per_tok = shifted_token_nll(ref_logits, labels)  # == -logp_ref(token)
    ref_per_tok = ref_per_tok.to(device=per_tok.device, dtype=per_tok.dtype)

    log_ratio = per_tok - ref_per_tok  # == logp_ref - logp_policy
    kl_per_tok = torch.exp(log_ratio) - log_ratio - 1.0  # k3 estimator of KL(policy || ref)
    kl = (kl_per_tok * mask).sum() / mask.sum().clamp(min=1)
    total = pg_loss + kl_beta * kl
    if return_metrics:
        return total, {"pg_loss": pg_loss.detach(), "kl": kl.detach()}
    return total


class WeightedSFTTrainer(CustomSeq2SeqTrainer):
    r"""Weight each example's NLL by its scalar advantage, with a KL trust region to a reference.

    The reference policy is the same LoRA model with the adapter disabled (``disable_adapter``),
    so the trust region is anchored to the base model. The reference forward runs under
    ``no_grad`` (~50% extra forward compute, no backward). ``kl_beta > 0`` is REQUIRED for
    stability with negative advantages — ``kl_beta = 0.0`` reproduces the divergence.

    Note: this overrides ``compute_loss`` entirely and recomputes cross-entropy from
    ``outputs.logits``, discarding ``outputs.loss`` and any auxiliary loss the model returns.
    That is fine for dense Qwen3. For a Qwen3-MoE variant the router load-balancing aux loss
    would be dropped — add ``outputs.aux_loss`` back here in that case.
    """

    def __init__(
        self,
        *args,
        weighted_reduction: str = "example_mean",
        kl_beta: float = 0.05,
        kl_ref_adapter_name: Optional[str] = None,
        model_adapter_name: str = "default",
        **kwargs,
    ) -> None:
        if weighted_reduction not in REDUCTION_MODES:
            raise ValueError(f"Unknown weighted_reduction {weighted_reduction!r}; expected one of {REDUCTION_MODES}.")
        if kl_beta < 0.0:
            raise ValueError(f"kl_beta must be >= 0, got {kl_beta!r}.")
        self.weighted_reduction = weighted_reduction
        self.kl_beta = kl_beta
        # Reference policy for the KL term:
        #   kl_ref_adapter_name is None -> base model (LoRA disabled). Correct when the starting
        #     adapter is fresh/zero or has been merged into the base you loaded.
        #   kl_ref_adapter_name set      -> activate that frozen adapter for the ref pass, i.e.
        #     reference = base + start-of-iteration adapter (the data-generating policy). This is
        #     the correct GRPO reference when continuing an adapter without merging.
        # model_adapter_name is the trainable adapter restored after each reference pass.
        self.kl_ref_adapter_name = kl_ref_adapter_name
        self.model_adapter_name = model_adapter_name
        # per-logging-interval accumulators for the loss components (flushed in `log`)
        self._weighted_metrics: dict[str, list[float]] = defaultdict(list)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # advantage is attached only to the TRAIN batch; during eval the collator defaults it
        # to ones, and if absent entirely we fall back to ones inside the loss helper.
        advantages = inputs.pop("advantage", None)

        outputs = model(**inputs)

        # Reference per-token NLL for the KL trust region. Skipped during eval (no advantage
        # column) and when kl_beta == 0, so eval takes no extra forward pass.
        ref_per_tok = None
        if self.kl_beta > 0.0 and advantages is not None:
            ref_per_tok = self._reference_per_tok(model, inputs)

        loss, metrics = compute_weighted_sft_loss(
            outputs.logits,
            inputs["labels"],
            advantages,
            reduction=self.weighted_reduction,
            ref_per_tok=ref_per_tok,
            kl_beta=self.kl_beta,
            return_metrics=True,
        )

        # Stash the components for wandb / logging. Train-only: skip eval so eval steps (which the
        # collator feeds advantages=ones) don't pollute the train pg_loss/kl curves.
        if model.training:
            for key, value in metrics.items():
                self._weighted_metrics[key].append(value.item())

        return (loss, outputs) if return_outputs else loss

    def _reference_per_tok(self, model, inputs):
        r"""Reference-policy per-token NLL ``[B, T-1]`` for the KL term, run under ``no_grad``.

        Two reference modes (see :meth:`__init__`). In both, we collapse the ``[B, T, V]``
        reference logits to the ``[B, T-1]`` NLL *inside* the ``no_grad`` block and free the
        logits, so the vocab-sized tensor never coexists with the policy backward graph (memory).
        """
        unwrapped = self.accelerator.unwrap_model(model)

        if self.kl_ref_adapter_name is None:
            # reference = base model: disable the LoRA adapter entirely.
            if not hasattr(unwrapped, "disable_adapter"):
                raise RuntimeError(
                    "kl_beta > 0 needs a PEFT/LoRA model to form the reference policy "
                    "(adapter disabled), but the unwrapped model has no disable_adapter(). "
                    "Use LoRA, or set kl_beta = 0 (UNSTABLE with negative advantages)."
                )
            with torch.no_grad(), unwrapped.disable_adapter():
                ref_logits = model(**inputs).logits
                ref_per_tok = shifted_token_nll(ref_logits, inputs["labels"])
                del ref_logits
            return ref_per_tok

        # reference = base + start-of-iteration adapter: activate the frozen reference adapter for
        # the ref forward, then restore the trainable adapter before the backward pass. set_adapter
        # also toggles requires_grad (frozen ref off, trainable on) — the restore in `finally` is
        # what re-enables grad on the policy adapter. Mirrors TRL's ref_adapter_name pattern.
        if not hasattr(unwrapped, "set_adapter"):
            raise RuntimeError(
                f"kl_ref_adapter_name={self.kl_ref_adapter_name!r} needs a PEFT model with "
                "set_adapter(), but the unwrapped model has none."
            )
        try:
            unwrapped.set_adapter(self.kl_ref_adapter_name)
            with torch.no_grad():
                ref_logits = model(**inputs).logits
                ref_per_tok = shifted_token_nll(ref_logits, inputs["labels"])
                del ref_logits
        finally:
            unwrapped.set_adapter(self.model_adapter_name)  # restore the trainable adapter
        return ref_per_tok

    def log(self, logs, *args, **kwargs):
        # Inject the mean pg_loss / kl over the logging interval next to HF's "loss" / "grad_norm",
        # so they show up as their own wandb panels. Cleared each interval to match HF's averaging.
        for key, values in self._weighted_metrics.items():
            if values:
                logs[key] = round(sum(values) / len(values), 6)
        self._weighted_metrics.clear()
        return super().log(logs, *args, **kwargs)


@dataclass
class WeightedCollator(SFTDataCollatorWith4DAttentionMask):
    r"""SFT collator that splices the per-example advantage into the batch.

    Eval-safe: features without an ``advantage`` key default to ``1.0``, so an eval dataset
    (which has no advantage column) collates into a batch of ones rather than raising.
    """

    def __call__(self, features):
        advs = [f.pop("advantage", 1.0) for f in features]
        batch = super().__call__(features)
        batch["advantage"] = torch.tensor(advs, dtype=torch.float32)
        return batch
