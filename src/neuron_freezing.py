"""Incremental neuron-freezing plasticity decay.

Ports the mechanism from Zhu & Warstadt, "Manipulating Plasticity Decay Shows
Critical Period Effects for L2 Acquisition in Neural Language Models" (the
``FreezeCallback`` of that paper), adapted to CLIMB-Sleep's wake/sleep trainer.

A "neuron" is one output feature of an MLP/attention projection: row ``i`` of a
2D weight matrix ``[out_features, in_features]`` together with element ``i`` of
the matching 1D bias. Neurons are frozen *permanently* by masking their
gradients: once frozen, a neuron's weight-row and bias-element gradients are
zeroed (``complete``) or scaled down (``partial``) on every backward pass.
Note: with a stateful optimizer (AdamW), a just-frozen row can still drift for
~tens of steps from residual momentum (exp_avg/exp_avg_sq) accumulated before
freezing; with weight_decay=0 it then settles and stays fixed. This matches the
paper's gradient-masking method; it is a "soft" freeze during the momentum tail,
not an instantaneous weight lock.

Unlike the paper's continuous, global-step schedule, freezing here is **tied to
the sleep cycle**: ``CustomTrainer`` calls :meth:`freeze_increment` once per
cycle (at SLEEP->WAKE) to grow the frozen set, and :meth:`mask_gradients` on
every backward step. A warmup (``freezing_ratio`` of total steps) keeps early
cycles fully plastic; the warmup gate lives in the trainer.

This is a plain object the trainer drives, *not* a ``TrainerCallback``: the
paper masks in ``on_substep_end``, which HF does not fire when
``gradient_accumulation_steps == 1`` (CLIMB-Sleep's effective setup). The
overridden ``training_step`` already runs right after backward and before the
base optimizer steps -- the robust seam.

Two selection modes:
  - ``random``: pick unfrozen rows uniformly at random (control).
  - ``causal``: rank unfrozen rows by Captum LayerIntegratedGradients
    attribution magnitude w.r.t. the LM loss; freeze the most causally active.
    ``captum`` is imported lazily so ``random`` runs need no extra dependency.
"""

import logging
import re
from typing import Dict, Optional, Set

import torch
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.functional import cross_entropy

logger = logging.getLogger(__name__)


class NeuronFreezer:
    """Grows a per-module frozen-neuron set each cycle and masks their grads.

    Args:
        base_model: the (unwrapped) base encoder whose neurons are frozen.
        head: the separate MLM head, needed to build the LM loss for ``causal``
            attribution (CLIMB-Sleep keeps the head out of ``base_model``).
        mode: ``"random"`` or ``"causal"``.
        freeze_target: ``"mlp"``, ``"attn"``, or ``"all"`` -- which Linear
            projections are freezable.
        first_m_layers: only consider transformer layers with index
            ``< first_m_layers`` (``<= 0`` disables the filter).
        n_layers: at each freeze event, restrict to a random subset of this many
            layers (``<= 0`` = all layers).
        freeze_type: ``"complete"`` (zero grads) or ``"partial"`` (scale grads
            by ``plasticity_decay_factor``).
        decay: delta -- fraction of each module's *remaining freezable pool* to
            freeze per cycle.
        plasticity_ceiling: c -- max fraction of a module's neurons that may ever
            be frozen.
        plasticity_decay_factor: grad scale for ``partial`` freezing.
        seed: RNG seed for ``random`` selection / layer subsetting.
        device: unused directly (frozen indices are built on each grad's device)
            but accepted for API symmetry with the trainer.
    """

    # Layer-index patterns. The RoBERTa pattern (singular ``.layer.N.``) is added
    # to the paper's GPT-2/LLaMA/T5 patterns for portability.
    _LAYER_PATTERNS = (
        r"\.layer\.(\d+)\.",
        r"\.layers\.(\d+)\.",
        r"\.h\.(\d+)\.",
        r"\.block\.(\d+)\.",
    )
    # Captum integration steps and a row cap to bound per-cycle attribution cost.
    _CAUSAL_N_STEPS = 16
    _CAUSAL_MAX_ROWS = 8

    def __init__(
        self,
        base_model: Module,
        head: Module,
        *,
        mode: str = "random",
        freeze_target: str = "mlp",
        first_m_layers: int = -1,
        n_layers: int = -1,
        freeze_type: str = "complete",
        decay: float = 0.1,
        plasticity_ceiling: float = 0.5,
        plasticity_decay_factor: float = 0.1,
        seed: int = 0,
        device=None,
    ) -> None:
        if mode not in {"random", "causal"}:
            raise ValueError(f"mode must be 'random' or 'causal', got {mode!r}")
        if freeze_target not in {"mlp", "attn", "all"}:
            raise ValueError(
                f"freeze_target must be 'mlp'|'attn'|'all', got {freeze_target!r}"
            )
        if freeze_type not in {"complete", "partial"}:
            raise ValueError(
                f"freeze_type must be 'complete' or 'partial', got {freeze_type!r}"
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if not 0.0 < plasticity_ceiling <= 1.0:
            raise ValueError(f"plasticity_ceiling must be in (0, 1], got {plasticity_ceiling}")
        if not 0.0 <= plasticity_decay_factor <= 1.0:
            raise ValueError(
                f"plasticity_decay_factor must be in [0, 1], got {plasticity_decay_factor}"
            )
        if mode == "causal":
            # Fail fast at construction rather than ~freezing_ratio into training.
            import importlib.util

            if importlib.util.find_spec("captum") is None:
                raise ImportError(
                    "mode='causal' requires captum (pip install captum)."
                )
            # Causal selection is data-dependent, so under DDP each rank would
            # attribute on its own shard and freeze a DIFFERENT neuron set,
            # silently diverging the replicas. Random mode is DDP-safe.
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    logger.warning(
                        "NeuronFreezer(causal) with world_size>1: each rank freezes a "
                        "different neuron set (replicas will diverge). Use mode='random' "
                        "for multi-GPU, or add a rank-0 broadcast of the frozen indices."
                    )
            except Exception:
                pass

        self.base_model = base_model
        self.head = head
        self.mode = mode
        self.freeze_target = freeze_target
        self.first_m_layers = int(first_m_layers)
        self.n_layers = int(n_layers)
        self.freeze_type = freeze_type
        self.decay = float(decay)
        self.plasticity_ceiling = float(plasticity_ceiling)
        self.plasticity_decay_factor = float(plasticity_decay_factor)

        import random

        self._rng = random.Random(seed)
        self._events = 0
        # Canonical, device-agnostic frozen sets: {module_name: {row indices}}.
        self.frozen: Dict[str, Set[int]] = {}
        # Lazily-built LongTensor view of each set on the grad's device.
        self._idx_cache: Dict[str, Tensor] = {}
        # {module_name: nn.Linear} freezable projections.
        self._targets: Dict[str, Linear] = self._candidate_modules()
        logger.info(
            "NeuronFreezer initialised: mode=%s target=%s freeze_type=%s "
            "decay=%.3f ceiling=%.3f modules=%d capacity=%d",
            mode,
            freeze_target,
            freeze_type,
            self.decay,
            self.plasticity_ceiling,
            len(self._targets),
            self.total_capacity(),
        )

    # ------------------------------------------------------------------ #
    # Module selection
    # ------------------------------------------------------------------ #
    def _extract_layer_number(self, name: str) -> int:
        for pat in self._LAYER_PATTERNS:
            m = re.search(pat, name)
            if m:
                return int(m.group(1))
        lname = name.lower()
        if any(k in lname for k in ("embed", "wte", "wpe")):
            return -1
        return 1000  # un-numbered (final norm / pooler); never a target here.

    def _is_target_name(self, name: str) -> bool:
        lname = name.lower()
        is_attn = "attention" in lname
        is_mlp = "intermediate.dense" in lname or (
            lname.endswith("output.dense") and not is_attn
        )
        if self.freeze_target == "mlp":
            return is_mlp
        if self.freeze_target == "attn":
            return is_attn
        return is_mlp or is_attn  # "all"

    def _candidate_modules(self) -> Dict[str, Linear]:
        targets: Dict[str, Linear] = {}
        for name, module in self.base_model.named_modules():
            if not isinstance(module, Linear):
                continue
            if not self._is_target_name(name):
                continue
            if self.first_m_layers > 0:
                layer = self._extract_layer_number(name)
                if layer >= self.first_m_layers:
                    continue
            targets[name] = module
        if not targets:
            logger.warning(
                "NeuronFreezer found no freezable modules for target=%s "
                "(check param naming / first_m_layers).",
                self.freeze_target,
            )
        return targets

    # ------------------------------------------------------------------ #
    # Public API driven by the trainer
    # ------------------------------------------------------------------ #
    def freeze_increment(self, inputs: Optional[Dict[str, Tensor]] = None) -> None:
        """Grow the frozen set by ``decay``-fraction of each module's remaining
        freezable pool (capped at ``ceiling``). One call per sleep cycle."""
        if self.mode == "causal" and inputs is None:
            logger.warning(
                "NeuronFreezer(causal): no cached batch available; "
                "skipping this freeze event."
            )
            return

        module_items = list(self._targets.items())
        if self.n_layers > 0:
            layers = sorted({self._extract_layer_number(n) for n, _ in module_items})
            self._rng.shuffle(layers)
            keep = set(layers[: self.n_layers])
            module_items = [
                (n, m)
                for n, m in module_items
                if self._extract_layer_number(n) in keep
            ]

        total_added = 0
        for name, module in module_items:
            n_neurons = module.weight.shape[0]
            cap = int(self.plasticity_ceiling * n_neurons)
            already = self.frozen.get(name, set())
            room = cap - len(already)
            if room <= 0:
                continue
            unfrozen = [i for i in range(n_neurons) if i not in already]
            if not unfrozen:
                continue
            n_add = int(self.decay * room)
            if n_add == 0:
                # Tail safeguard: advance by ONE neuron rather than dumping the
                # whole remaining pool, so the schedule stays gradual and
                # independent of module width. (decay > 0 guaranteed by __init__.)
                # NOTE: this deviates from the paper's "freeze the remainder"
                # safeguard, which would collapse a near-exhausted module to its
                # ceiling in a single cycle.
                n_add = 1
            n_add = min(n_add, len(unfrozen))
            if n_add <= 0:
                continue

            if self.mode == "random":
                chosen = self._rng.sample(unfrozen, n_add)
            else:
                scores = self._causal_scores(module, inputs)
                if not torch.isfinite(scores).any():
                    logger.warning(
                        "NeuronFreezer(causal): non-finite attributions for %s; "
                        "skipping this module.", name,
                    )
                    continue
                # NaN/inf sort highest in topk; exclude non-finite scores so a
                # degenerate attribution can never be selected for freezing.
                scores = torch.nan_to_num(
                    scores.clone(), nan=float("-inf"),
                    posinf=float("-inf"), neginf=float("-inf"),
                )
                if already:
                    scores[list(already)] = float("-inf")
                chosen = torch.topk(scores, n_add).indices.tolist()

            frozen_set = self.frozen.setdefault(name, set())
            frozen_set.update(int(i) for i in chosen)
            self._idx_cache.pop(name, None)  # invalidate cached LongTensor
            total_added += len(chosen)

        self._events += 1
        logger.info(
            "NeuronFreezer(%s) event %d: froze +%d neurons (total %d / cap %d).",
            self.mode,
            self._events,
            total_added,
            self.num_frozen(),
            self.total_capacity(),
        )

    def mask_gradients(self, base_model: Optional[Module] = None) -> None:
        """Zero (or scale) the gradients of all frozen neuron rows. Called every
        backward step. ``base_model`` is accepted for API symmetry; the stored
        module references already point at the live parameters."""
        if not self.frozen:
            return
        for name, module in self._targets.items():
            rows = self.frozen.get(name)
            if not rows:
                continue
            weight = module.weight
            if weight.grad is None:
                continue
            idx = self._idx_cache.get(name)
            if idx is None or idx.device != weight.grad.device:
                idx = torch.tensor(
                    sorted(rows), dtype=torch.long, device=weight.grad.device
                )
                self._idx_cache[name] = idx
            self._mask_one(weight, idx)
            bias = module.bias
            if bias is not None and bias.grad is not None:
                self._mask_one(bias, idx)

    def _mask_one(self, param, idx: Tensor) -> None:
        grad = param.grad
        if self.freeze_type == "complete":
            grad.index_fill_(0, idx, 0.0)
        else:  # partial: shrink the frozen rows' gradients by a constant factor.
            grad.index_copy_(
                0, idx, grad.index_select(0, idx) * self.plasticity_decay_factor
            )

    # ------------------------------------------------------------------ #
    # Causal (Captum LayerIntegratedGradients) scoring
    # ------------------------------------------------------------------ #
    def _causal_scores(self, module: Linear, inputs: Dict[str, Tensor]) -> Tensor:
        """Per-neuron |attribution| of the LM loss to ``module``'s output."""
        from captum.attr import LayerIntegratedGradients

        input_ids = inputs["input_ids"][: self._CAUSAL_MAX_ROWS]
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn[: self._CAUSAL_MAX_ROWS]
        labels = inputs["labels"][: self._CAUSAL_MAX_ROWS]

        emb_layer = self.base_model.get_input_embeddings()
        inputs_embeds = emb_layer(input_ids).detach().requires_grad_(True)
        base_model, head = self.base_model, self.head

        def forward_fn(embeds, attention_mask, lbl):
            hidden = base_model(
                inputs_embeds=embeds, attention_mask=attention_mask
            )[0]
            logits = head(hidden).transpose(-1, -2)  # [B, V, S]
            per_tok = cross_entropy(logits, lbl, reduction="none")  # [B, S]
            mask = (lbl != -100).float()
            return (per_tok * mask).sum(-1) / mask.sum(-1).clamp(min=1)  # [B]

        was_training = base_model.training
        base_model.eval()
        head.eval()
        try:
            lig = LayerIntegratedGradients(forward_fn, module)
            attr = lig.attribute(
                inputs_embeds,
                baselines=torch.zeros_like(inputs_embeds),
                additional_forward_args=(attn, labels),
                n_steps=self._CAUSAL_N_STEPS,
            )  # [B, S, F] -- F = module.out_features (one per neuron)
        finally:
            if was_training:
                base_model.train()
                head.train()
        return attr.abs().sum(dim=(0, 1))  # [F]

    # ------------------------------------------------------------------ #
    # Introspection / logging helpers
    # ------------------------------------------------------------------ #
    def num_frozen(self) -> int:
        return sum(len(v) for v in self.frozen.values())

    def total_capacity(self) -> int:
        return sum(
            int(self.plasticity_ceiling * m.weight.shape[0])
            for m in self._targets.values()
        )

    def ceiling_reached(self) -> bool:
        """True once every module is at its per-module ceiling. Logging only --
        the trainer never stops on this (unlike the paper's c==1 stop)."""
        for name, module in self._targets.items():
            cap = int(self.plasticity_ceiling * module.weight.shape[0])
            if len(self.frozen.get(name, ())) < cap:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Checkpoint persistence (the frozen set is NOT reconstructable for
    # causal mode, and must survive the cluster's resume-from-checkpoint).
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        return {
            "frozen": {name: sorted(rows) for name, rows in self.frozen.items()},
            "events": self._events,
            "rng": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.frozen = {name: set(rows) for name, rows in state.get("frozen", {}).items()}
        self._events = state.get("events", 0)
        if state.get("rng") is not None:
            # getstate() round-trips through pickle as nested lists; setstate
            # needs the inner state to be a tuple.
            v, internal, gauss = state["rng"]
            self._rng.setstate((v, tuple(internal), gauss))
        self._idx_cache.clear()
