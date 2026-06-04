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

r"""Entry point for advantage-weighted SFT (offline GRPO).

This is our own driver — we do not go through ``llamafactory-cli``. It reuses LlamaFactory's
argument parsing, tokenizer/model/dataset loading, then swaps in :class:`WeightedSFTTrainer`
and :class:`WeightedCollator` and splices a per-example ``advantage`` column onto the train
dataset after tokenization.

Run with::

    python scripts/train_weighted.py train.yaml data/your_dataset.json

where ``train.yaml`` uses the same keys as a LlamaFactory SFT config, plus two optional keys
we consume here before parsing:

- ``weighted_reduction``: one of batch_token_mean / example_mean / sum (default example_mean).
- ``kl_beta``: KL trust-region coefficient (default 0.05). ``0.0`` disables the trust region
  and is UNSTABLE with negative advantages — the loss runs away. Also set ``max_grad_norm: 1.0``
  and a conservative learning rate (~1e-5) in the YAML; the trust region is what keeps the
  negative-advantage gradient bounded.
- ``kl_reference``: which reference the KL is measured against (default ``start_adapter`` when an
  adapter is being resumed, else ``base``).

  - ``base``: bare base model (LoRA disabled). Correct only when the starting adapter is fresh /
    zero or has been merged into the base you loaded.
  - ``start_adapter``: base + the adapter as loaded at the start of this run, i.e. the
    data-generating (rollout) policy. This is the correct GRPO reference when you continue an
    adapter across iterations WITHOUT merging it into the base. Requires ``adapter_name_or_path``.

The JSON file is the data file referenced by your ``dataset_info`` entry.
"""

import json
import sys

import yaml

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.sft.weighted import WeightedCollator, WeightedSFTTrainer


def load_advantages_in_order(dataset_json_path: str) -> list[float]:
    r"""Read advantages from the SAME JSON LlamaFactory will load, in file order.

    Reading the same file in the same order is what guarantees alignment with the tokenized
    dataset rows (LlamaFactory preserves row order through tokenization when packing is off).
    """
    with open(dataset_json_path, encoding="utf-8") as f:
        rows = json.load(f)
    return [float(r["advantage"]) for r in rows]


def run_weighted_sft(config: dict, advantages_in_order: list[float]) -> None:
    # `weighted_reduction`, `kl_beta`, `kl_reference` are our own keys; strip them before
    # LlamaFactory's parser sees the config (it would reject unknown fields).
    weighted_reduction = config.pop("weighted_reduction", "example_mean")
    kl_beta = float(config.pop("kl_beta", 0.05))
    kl_reference = config.pop("kl_reference", None)  # None -> auto (start_adapter if resuming)

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)

    # per-example advantage is undefined if multiple examples are packed into one sequence
    assert not getattr(data_args, "packing", False), "Disable packing for weighted SFT."
    assert not getattr(data_args, "neat_packing", False), "Disable neat_packing for weighted SFT."

    tok_module = load_tokenizer(model_args)
    tokenizer = tok_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    ds_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tok_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # Resolve the KL reference. Default: if we are resuming an adapter (not merged into base) and
    # the trust region is on, anchor to that start-of-iteration adapter; otherwise to base.
    resuming_adapter = bool(model_args.adapter_name_or_path) and not finetuning_args.create_new_adapter
    if kl_reference is None:
        kl_reference = "start_adapter" if (kl_beta > 0.0 and resuming_adapter) else "base"
    assert kl_reference in ("base", "start_adapter"), f"kl_reference must be base|start_adapter, got {kl_reference!r}."

    kl_ref_adapter_name = None
    if kl_beta > 0.0 and kl_reference == "start_adapter":
        assert model_args.adapter_name_or_path, (
            "kl_reference: start_adapter needs a loaded adapter (adapter_name_or_path) to anchor "
            "to. For a fresh adapter the start policy IS the base model — use kl_reference: base."
        )
        # Load the resumed adapter (the last one; earlier adapters are already merged into base by
        # LlamaFactory) a SECOND time, frozen, as the reference. base + this == the rollout policy.
        kl_ref_adapter_name = "kl_reference"
        model.load_adapter(model_args.adapter_name_or_path[-1], adapter_name=kl_ref_adapter_name, is_trainable=False)
        model.set_adapter("default")  # keep training the trainable ("default") adapter

    train_ds = ds_module["train_dataset"]
    assert len(train_ds) == len(advantages_in_order), (
        f"misaligned: {len(train_ds)} rows vs {len(advantages_in_order)} advantages — "
        "length filtering likely dropped rows; raise cutoff_len."
    )
    train_ds = train_ds.add_column("advantage", advantages_in_order)
    ds_module["train_dataset"] = train_ds

    # Build the collator the same way run_sft does, so 4D masks / dtypes / padding match upstream.
    collator = WeightedCollator(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tok_module,
    )

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=collator,
        weighted_reduction=weighted_reduction,
        kl_beta=kl_beta,
        kl_ref_adapter_name=kl_ref_adapter_name,
        **ds_module,
        **tok_module,
    )
    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    config_path = sys.argv[1]  # training YAML (same keys as a llamafactory sft config)
    dataset_json = sys.argv[2]  # the data file referenced by your dataset_info entry
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    advantages = load_advantages_in_order(dataset_json)
    run_weighted_sft(config, advantages)
