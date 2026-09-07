""" Main trainer class for BabyLM. """

import copy
import json
import logging
import os
import time
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import AdamW

# Data loading
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Model Training
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    Trainer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_utils import PreTrainedModel, unwrap_model

# from transformers.utils import (
#     get_full_repo_name,
#     is_torch_neuroncore_available,
# )
from transformers.models.roberta_prelayernorm.modeling_roberta_prelayernorm import (
    RobertaPreLayerNormLMHead,
)
from transformers.trainer_utils import (
    IntervalStrategy,
    has_length,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import is_torch_neuroncore_available
from wandb import Table

from src.data_curriculum.utility_scoring import compute_need_scores

# Model Loading
from src.models import build_inference_lm, load_base_model
from src.synaptic_homeostasis import (
    apply_fisher_protected_shrink,
    taylor_saliency,
)
from src.utils.data import base_collate_fn
from src.utils.inference import compute_trainer_perplexity
from src.utils.replay_score import per_sample_mean_token_loss
from src.utils.per_sample_grad import (
    per_sample_grads,
    per_sample_squared_grad_norms,
)

# typing imports
from .config import BabyLMConfig

# Sleep Sampling
from .data_curriculum.sleep_sampler import SleepSampler

# Sleep Data Loader
from .dataloader import SleepDataLoader

# Model Evaluation
from .evaluator import BabyLMEvaluator, BlimpEvaluator, FinetuneEvaluator

# from huggingface_hub.repository import Repository
# from huggingface_hub.utils._errors import HfHubHTTPError


logger = logging.getLogger(__name__)


class CustomTrainer(Trainer):
    def __init__(
        self,
        hydra_config: BabyLMConfig,
        dry_run: bool,
        args: TrainingArguments,
        tokenizer: PreTrainedTokenizerFast,
        sleep_table: Table,
        max_steps_per_phase: Dict[str, int],
        **kwargs,
    ) -> None:
        """
        We need to override the __init__ method to add the experiment group and experiment name.

        We use the group name and experiment name for version controlling/identifying the current
        run in, for example, huggingface, wandb ...

        Args:
            * hydra_config: (BabyLMConfig): The config object.
            * dry_run (bool): Whether the experiment is being run in dry run mode
            * args (TrainingArguments): The training arguments, unpacked from the kwargs dict
                in order to have access to possible arguments meant to be used in the Custom
                Trainer class.
            * tokenizer (PreTrainedTokenizerFast): The tokenizer used for the current run.
        """

        self.hydra_config = hydra_config
        self.dry_run = dry_run
        self.control = None

        self.experiment_group = hydra_config.experiment.group
        self.experiment_name = hydra_config.experiment.name
        self.eval_blimp = hydra_config.trainer.eval_blimp
        self.eval_fast = hydra_config.trainer.eval_fast
        self.eval_glue = hydra_config.trainer.eval_glue
        self.eval_msgs = hydra_config.trainer.eval_msgs
        self.eval_perplexity = hydra_config.trainer.eval_perplexity
        self.sleep_table = sleep_table
        self.task_name = hydra_config.task.task

        super().__init__(args=args, **kwargs)

        self.sleep_mechanism_cfg = hydra_config.sleep_mechanism
        self.max_steps_per_phase = max_steps_per_phase

        # Plasticity decay state (Method 2: online empirical Fisher protection).
        # Synaptic-Intelligence-style: Fisher is accumulated during wake at the
        # moving theta of training, not at a fixed end-of-wake snapshot. This is
        # a deliberate deviation from strict EWC; SI (Zenke 2017) is the citation.
        self._plasticity_decay_enabled = bool(
            self.sleep_mechanism_cfg
            and self.sleep_mechanism_cfg.plasticity_decay is not None
            and self.sleep_mechanism_cfg.plasticity_decay.enabled
        )
        # Fisher accumulator built during wake; finalized at WAKE->SLEEP.
        self.fisher_accumulator: Optional[Dict[str, torch.Tensor]] = None
        self.fisher_sample_count = 0
        # Finalized importance scores, consumed by the SLEEP->WAKE shrink.
        # Last cycle's scores are never consumed (no SLEEP->WAKE after the
        # final WAKE->SLEEP); held memory is reclaimed at process exit.
        self.fisher_diagonal: Optional[Dict[str, torch.Tensor]] = None

        # The Taylor signals replace the wake-trajectory accumulation with a
        # single scoring pass at the end-of-wake weights, so they need the last
        # wake batch kept around; see _finalize_taylor_saliency.
        self._importance_signal = (
            self.sleep_mechanism_cfg.plasticity_decay.importance_signal
            if self._plasticity_decay_enabled
            else "fisher"
        )
        if self._importance_signal not in (
            "fisher",
            "taylor_signed",
            "taylor_abs",
        ):
            raise ValueError(
                "plasticity_decay.importance_signal must be one of 'fisher', "
                "'taylor_signed', 'taylor_abs'; got "
                f"{self._importance_signal!r}"
            )
        self._taylor_signal_enabled = self._plasticity_decay_enabled and (
            self._importance_signal != "fisher"
        )
        self._last_wake_batch: Optional[Dict[str, torch.Tensor]] = None

        # Wall-clock accounting (reviewer-requested GPU-hour reporting). Phase
        # time excludes the eval that runs at each switch. SH costs are
        # cumulative seconds across the run so the overhead of a signal can be
        # read off WandB directly: Fisher pays per_sample_grad on every wake
        # step, Taylor pays one scoring pass per cycle.
        self._train_wall_start: Optional[float] = None
        self._phase_wall_start: Optional[float] = None
        self._sh_time_sec = {"per_sample_grad": 0.0, "score": 0.0, "shrink": 0.0}

        self.phase_steps = 0

        # NOTE: The hidden dimension of the base model is the input dimension to the task head
        # We check that this variable is set in the config file when loading the base model

        self.hidden_rep_size = hydra_config.model.model_kwargs["hidden_size"]
        if self.sleep_mechanism_cfg:
            logger.info(
                f"Using sleep mechanism configuration {self.sleep_mechanism_cfg}"
            )

        self.tokenizer = tokenizer

        # track gradient steps for checkpointing
        self.global_step = 0

    def _get_train_sampler(self):
        """
        Overriding this method to use custom samplers that enable sleep mechanism.
        """

        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = (
            self.args.data_seed
            if self.args.data_seed is not None
            else self.args.seed
        )

        # Save sampler in an attribute
        if not hasattr(self, "_train_sampler"):
            if self.sleep_mechanism_cfg:
                # Sleep-consolidated learning: use SleepSampler for wake/sleep cycles
                logger.info(
                    "Using SleepSampler for sleep-consolidated learning"
                )

                self._train_sampler = SleepSampler(
                    dataset=self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    replay_ratio=self.sleep_mechanism_cfg.replay_ratio,
                    n_phases=self.sleep_mechanism_cfg.n_phases,
                    max_seq_length=self.sleep_mechanism_cfg.max_seq_length,
                    decay_rate=self.sleep_mechanism_cfg.replay_decay_rate,
                    min_decay_factor=self.sleep_mechanism_cfg.min_decay_factor,
                    contextualize_sleep=self.sleep_mechanism_cfg.contextualize_sleep,
                    n_augmentations=self.sleep_mechanism_cfg.n_augmentations,
                    replay_strategy=self.sleep_mechanism_cfg.replay_strategy,
                    utility_temperature_gain=self.sleep_mechanism_cfg.utility_temperature_gain,
                    utility_temperature_need=self.sleep_mechanism_cfg.utility_temperature_need,
                )
            else:
                # We are not using the sleep mechanism, so we can use the default sampler.
                if self.args.world_size <= 1:
                    self._train_sampler = RandomSampler(self.train_dataset, generator=generator)  # type: ignore
                else:
                    self._train_sampler = DistributedSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
        return self._train_sampler

    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # if self.state.epoch is not None:
        #     logs["epoch"] = round(self.state.epoch, 2)
        logger.info(f"LOGGING... {logs}")
        logger.info(
            f"GLOBAL STEP: {self.state.global_step};{self.global_step}"
        )

        output = {**logs, **{"step": self.state.global_step}}
        if "sleep_table" not in logs:
            # NOTE: Everything in state will be serialized to a json file when we save
            # a checkpoint - alas a wandb.Table is not
            self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        The dataset is sorted by the scoring function, if provided.

        Returns:
            * (CustomDataLoader): The custom training dataloader, a subclass instance of the torch
                Dataloader.
        """

        assert self.train_dataset is not None

        # NOTE: The standard Trainer.get_train_dataloader() method removes unused columns for
        # training, we only remove the filename column here.
        # The other columns in the datast should now be either of type float or int
        # (filename is the only str column).
        # We will remove the other columns in a postprocessing step (after the objective
        # function collation).

        train_sampler = self._get_train_sampler()

        train_dataset = self.train_dataset.remove_columns("filename")  # type: ignore

        # check if using sleep mechanism w/ SleepSampler
        if self.sleep_mechanism_cfg and isinstance(
            train_sampler, SleepSampler
        ):
            return SleepDataLoader(
                dataset=train_dataset,
                sampler=train_sampler,
                config=self.hydra_config,
                tokenizer=self.tokenizer,
                batch_size=self._train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                num_workers=0,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return DataLoader(
                dataset=train_dataset,
                sampler=train_sampler,
                batch_size=self._train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                num_workers=0,
                pin_memory=self.args.dataloader_pin_memory,
            )

        assert (
            self.tokenizer is not None
        ), "Tokenizer is not set. Please set the tokenizer before calling the train method."

    def compute_loss(
        self,
        model,
        inputs,
        override_input_ids=None,
        override_labels=None,
        loss_kwargs=None,
        inference=False,
        return_outputs=False,
        **kwargs,
    ):
        """
        ~~We compute the loss for each objective unit, and then sum them up.~~
        We track the loss of each sample during WAKE phases to add to the replay buffer.
        """

        loss_metrics = {}

        if not inference and self.state.global_step >= self.args.max_steps:
            raise Exception(
                """
                Reached max_steps already - training should have stopped.
                NOTE: You are probably using a resume_from_checkpoint flag with max_steps set to a
                value smaller than the number of steps in the checkpoint.
                """
            )

        # Check if we should track per-sample losses for sleep mechanism
        track_per_sample = (
            self.sleep_mechanism_cfg is not None
            and not inference
            and hasattr(self, "callback_handler")
            and hasattr(self.callback_handler, "train_dataloader")
            and self.callback_handler.train_dataloader is not None
            and hasattr(self.callback_handler.train_dataloader, "sampler")
            and hasattr(
                self.callback_handler.train_dataloader.sampler, "phase"
            )
            and self.callback_handler.train_dataloader.sampler.phase == "WAKE"
        )
        input_ids = (
            override_input_ids
            if override_input_ids is not None
            else inputs["input_ids"]
        )
        outputs = model(
            input_ids=input_ids,
            attention_mask=(
                inputs["attention_mask"]
                if "attention_mask" in inputs
                else None
            ),
        )
        # base_model_hidden_states = outputs[0]
        logits = outputs[0].transpose(-1, -2)

        labels = (
            override_labels
            if override_labels is not None
            else inputs["labels"]
        )

        if self.task_name == "clm":
            logits = logits[:, :, :-1].contiguous()
            labels = labels[:, 1:].contiguous()

        # Compute the loss
        if track_per_sample:
            # Per-sample squared gradient norms (Gain) for Gain x Need replay
            # AND/OR online empirical Fisher accumulation for plasticity decay.
            # One torch.func.vmap pass produces grads consumed by both methods.
            # The Taylor signals score once at the phase boundary, so they do
            # NOT need the per-sample grads on every wake step -- that would be
            # a full vmap pass per step thrown away.
            need_per_sample_grads = (
                self.sleep_mechanism_cfg.replay_criteria == "utility"
                or (
                    self._plasticity_decay_enabled
                    and not self._taylor_signal_enabled
                )
            )
            if need_per_sample_grads:
                _t0 = time.time()
                grads_dict = per_sample_grads(
                    unwrap_model(model),
                    input_ids,
                    inputs["attention_mask"],
                    labels,
                    self.task_name,
                )
                self._sh_time_sec["per_sample_grad"] += time.time() - _t0
                if (
                    self._plasticity_decay_enabled
                    and not self._taylor_signal_enabled
                ):
                    self._accumulate_fisher(grads_dict)

            if self._taylor_signal_enabled:
                # The Taylor signals are scored once, at the end-of-wake
                # weights, rather than accumulated along the wake trajectory.
                # _finalize_fisher runs at the phase boundary where no batch is
                # in scope, so keep the most recent one. Detached and kept on
                # device; one batch, so the memory cost is negligible.
                self._last_wake_batch = {
                    "input_ids": input_ids.detach(),
                    "attention_mask": inputs["attention_mask"].detach(),
                    "labels": labels.detach(),
                }

            # Per-sample replay score for replay buffer tracking.
            per_sample_score = None
            if self.sleep_mechanism_cfg.replay_criteria == "loss":
                # Score = cross entropy loss
                per_sample_score = per_sample_mean_token_loss(
                    logits, labels, **(loss_kwargs or {})
                )
            elif self.sleep_mechanism_cfg.replay_criteria == "utility":
                # Score = Utility = gain * need
                # During wake phase, scores are stored as just gain
                # Need scores are calculated at the end of each wake phase
                per_sample_score = per_sample_squared_grad_norms(grads_dict)

            # Add per-sample loss (and Gain when computed) to sampler
            if "indices" in inputs:
                indices = inputs["indices"].tolist()
                scores = per_sample_score.detach().cpu().tolist()
                self.callback_handler.train_dataloader.sampler.add_to_candidates(
                    indices, scores
                )
        loss = cross_entropy(logits, labels, **(loss_kwargs or {}))
        if inference:  # Return loss early if inference
            return loss
        # averaging over the processes
        total_unit_loss_scalar = self._nested_gather(loss).mean().item()  # type: ignore
        loss_metrics[f"loss_{self.task_name}"] = total_unit_loss_scalar
        if self.sleep_mechanism_cfg:
            curr_phase = self.callback_handler.train_dataloader.sampler.phase
            loss_metrics[f"loss_{self.task_name}_{curr_phase}"] = (
                total_unit_loss_scalar
            )

        # increment step after each loss computation
        # compute_loss() runs once per batch during training (right when model does gradient update)
        # incrementing here ensures we track one step per gradient update
        self.global_step += 1
        self.phase_steps += 1

        if (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.args.logging_steps == 0
        ):

            self.log(loss_metrics)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, *args):
        loss = super().training_step(model, inputs)

        # == SLEEP MECHANISM == #
        if self.sleep_mechanism_cfg:
            # if wake phase over, reset phase steps, eval, then switch
            sampler = self.callback_handler.train_dataloader.sampler
            phase = sampler.phase
            phase_max = self.max_steps_per_phase[phase]

            if self.phase_steps >= phase_max:
                if phase == "WAKE" and (
                    self.sleep_mechanism_cfg.wake_block_steps
                    > sampler.wake_max_steps
                ):
                    logger.info(
                        "WARNING: The selected WAKE phase max steps exceeds the size of the current fold. Ending WAKE phase early."
                    )
                next_phase = "SLEEP" if phase == "WAKE" else "WAKE"
                self._swap_phase(sampler, phase, next_phase)

        return loss

    def evaluate(
        self,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Override the Trainer.evaluate() method to evaluate on BLIMP using the evaluation pipeline submodule.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        # NOTE: This code runs on all processes (i.e. multiple GPUs) in a distributed settings.

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()

        is_best_run = "best" in metric_key_prefix

        metrics = {}

        # Additional behavior - evaluate perplexity
        # Get 10_000 samples from the eval dataset
        eval_subset = self.eval_dataset.select(  # type: ignore
            range(
                self.args.process_index,  # local process rank
                self.eval_dataset.num_rows,  # type: ignore
                self.eval_dataset.num_rows // ((100 if self.dry_run else self.hydra_config.trainer.n_eval_samples) // self.args.world_size),  # type: ignore
            )
        )
        logging.info("Evaluating perplexity...")
        logging.info(f" ** Number of samples: {eval_subset.num_rows}")
        pad_idx = self.tokenizer.pad_token_id
        mask_idx = self.tokenizer.mask_token_id

        assert pad_idx is not None and mask_idx is not None

        if self.eval_perplexity:
            perplexities = []
            with torch.no_grad():

                inference_dataloader = DataLoader(
                    eval_subset,  # type: ignore
                    batch_size=self.hydra_config.trainer.eval_batch_size,
                    shuffle=False,
                    collate_fn=base_collate_fn,
                    pin_memory=True,
                )

                for batch in tqdm(inference_dataloader):
                    batch_perplexity = compute_trainer_perplexity(
                        batch, self.tokenizer, self
                    )

                    perplexities.extend(batch_perplexity)

            tensor_perplexities = torch.tensor(
                perplexities, device=self.args.device
            )
            perplexity_mean = torch.mean(tensor_perplexities)
            perplexity_std = torch.std(tensor_perplexities)
            perplexity_max = torch.max(tensor_perplexities)

            if self.args.world_size > 1:
                # setup barrier for all processes
                dist.barrier()

                # Reduce perplexity across all processes
                gathered_perplexity_mean = [
                    torch.zeros_like(perplexity_mean)
                    for _ in range(self.args.world_size)
                ]
                gathered_perplexity_std = [
                    torch.zeros_like(perplexity_std)
                    for _ in range(self.args.world_size)
                ]
                gathered_perplexity_max = [
                    torch.zeros_like(perplexity_max)
                    for _ in range(self.args.world_size)
                ]

                dist.all_gather(gathered_perplexity_mean, perplexity_mean)
                dist.all_gather(gathered_perplexity_std, perplexity_std)
                dist.all_gather(gathered_perplexity_max, perplexity_max)

            # if main process
            metrics[f"{metric_key_prefix}_perplexity_mean"] = torch.mean(
                torch.tensor(perplexities)
            ).item()
            metrics[f"{metric_key_prefix}_perplexity_std"] = torch.std(
                torch.tensor(perplexities)
            ).item()
            metrics[f"{metric_key_prefix}_perplexity_max"] = torch.max(
                torch.tensor(perplexities)
            ).item()

        self.save_model(self.args.output_dir, _internal_call=True)
        # if world size > 1, then we need to synchronize the model across all processes
        if self.args.world_size > 1:
            dist.barrier()  # Ensure all processes have access to the same model

        evaluator_metrics = {}

        inference_model_dir = os.path.join(self.args.output_dir, "lm_model")

        # Additional behaviour - evaluate on BLIMP
        if self.eval_blimp:
            logging.info("Evaluating on BLIMP and AOA...")
            blimp_evaluator = BlimpEvaluator(
                inference_model_dir,
                device=self.args.device,
                process_index=self.args.process_index,  # world (global) process index
                world_size=self.args.world_size,
                dry_run=self.dry_run,
                keep_predictions=is_best_run,
            )
            # Get average of blimp metrics
            blimp_metrics = blimp_evaluator()
            evaluator_metrics.update(blimp_metrics)  # type: ignore

        # BabyLM fast eval
        if self.eval_fast:
            logging.info("Evaluating on BabyLM Fast Evaluation...")
            babylm_evaluator = BabyLMEvaluator(
                inference_model_dir,
                device=self.args.device,
                process_index=self.args.process_index,  # world (global) process index
                world_size=self.args.world_size,
                dry_run=self.dry_run,
                keep_predictions=is_best_run,
                task="mlm" if self.task_name == "mlm" else "causal",
            )
            # Get average of blimp metrics
            babylm_metrics = babylm_evaluator()
            evaluator_metrics.update(babylm_metrics)  # type: ignore

        if self.eval_glue or self.eval_msgs:
            logging.info("Evaluating on finetuning tasks...")
            finetune_evaluator = FinetuneEvaluator(
                inference_model_dir,
                device=self.args.device,
                process_index=self.args.process_index,  # world (global) process index
                world_size=self.args.world_size,
                dry_run=self.dry_run,
                run_glue=self.eval_glue,
                run_msgs=self.eval_msgs,
                keep_predictions=is_best_run,
            )
            # Get average of glue metrics
            finetune_metrics = finetune_evaluator()
            evaluator_metrics.update(finetune_metrics)  # type: ignore

        for key in list(evaluator_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                evaluator_metrics[f"{metric_key_prefix}_{key}"] = (
                    evaluator_metrics.pop(key)
                )

        metrics.update(evaluator_metrics)

        if f"{metric_key_prefix}_jit_compilation_time" in metrics:
            start_time += metrics[f"{metric_key_prefix}_jit_compilation_time"]
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
            )
        )

        # Log step of best model if running final evaluation
        if is_best_run:
            metrics[f"{metric_key_prefix}_model_step"] = int(
                self.state.best_model_checkpoint.split("checkpoint-")[-1]
            )

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def _initialize_full_lm_model(self):
        """
        Initialize a full language model (base model + LM head) carrying the
        trained weights, for export to lm_model/.
        """
        return build_inference_lm(
            copy.deepcopy(self.hydra_config), unwrap_model(self.model)
        )

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Override the Trainer._save() method to save the objective curriculum state as well,
        and to save the full language model (base model + mlm head).
        """

        if self.args.should_save:
            super()._save(output_dir=output_dir, state_dict=state_dict)

            # Saving should be done only on the main process

            output_dir = (
                output_dir if output_dir is not None else self.args.output_dir
            )

            # save step count
            step_file = os.path.join(output_dir, "trainer_state.json")
            # write current step count to file inside checkpoint folder
            with open(step_file, "w") as f:
                json.dump({"global_step": self.global_step}, f)

            lm_model_dir = os.path.join(output_dir, "lm_model")
            # task_heads_dir = os.path.join(output_dir, "task_heads")
            os.makedirs(lm_model_dir, exist_ok=True)
            # os.makedirs(task_heads_dir, exist_ok=True)

            # save the full language model + the associated tokenizer (for inference)
            lm_model = self._initialize_full_lm_model()
            lm_model.save_pretrained(lm_model_dir)

            self.tokenizer.save_pretrained(lm_model_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        # load step count
        step_file = os.path.join(resume_from_checkpoint, "trainer_state.json")
        # reads from JSON file and restores step count
        if os.path.exists(step_file):
            with open(step_file, "r") as f:
                state = json.load(f)
                self.global_step = state.get("global_step", 0)

    def _wrap_model(self, model):
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs["find_unused_parameters"] = (
                    self.args.ddp_find_unused_parameters
                )
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs["find_unused_parameters"] = (
                    not model.is_gradient_checkpointing
                )
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            if is_torch_neuroncore_available():
                return model
            if any(p.requires_grad for p in model.parameters()):
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=(
                        [self.args.local_rank]
                        if self.args._n_gpu != 0
                        else None
                    ),
                    output_device=(
                        self.args.local_rank if self.args._n_gpu != 0 else None
                    ),
                    broadcast_buffers=False,
                    **kwargs,
                )

        return model

    def train(self, *args, resume_from_checkpoint=None, **kwargs):
        """
        Override train to re-intialize phase steps.
        """
        self.phase_steps = 0
        self._train_wall_start = time.time()
        self._phase_wall_start = self._train_wall_start
        try:
            return super().train(
                resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs
            )  # CHANGED from super().super().train to super().train
        finally:
            # The final WAKE->SLEEP finalizes a Fisher diagonal that is never
            # consumed (no SLEEP->WAKE follows). Release it explicitly so the
            # ~57 MB it occupies for the default model is not held until exit.
            self.fisher_diagonal = None
            self.fisher_accumulator = None
            self.fisher_sample_count = 0
            self._last_wake_batch = None

    def _swap_phase(self, sampler: SleepSampler, phase: str, next_phase: str):
        logger.info("Ending %s phase...", phase)
        phase_wall_sec = (
            time.time() - self._phase_wall_start
            if self._phase_wall_start is not None
            else float("nan")
        )
        self.phase_steps = 0
        # eval before sleep
        logger.info("Running evaluation before %s phase...", next_phase)
        self.evaluate(metric_key_prefix=f"eval_before_{next_phase}")

        if next_phase == "WAKE":
            # On switching to wake phase, build and log the sleep table
            logger.info("Logging replay samples...")

            samples, losses = list(zip(*sampler.get_replay_samples(10)))
            sample_ids = [samp["input_ids"] for samp in samples]
            sample_texts = "\n\n".join(
                [
                    f"{i}:\n - score: {round(losses[i][0], 2)}\n - factor: {round(losses[i][1], 2)}\n - text: {self.tokenizer.decode(ids)}"
                    for i, ids in enumerate(sample_ids)
                ]
            )

            self.sleep_table.add_data(
                self.global_step,  # global_step
                self.phase_steps,  # phase_step
                sampler.curr_fold,  # phase_num
                sample_texts,  # replay_samples
                # losses # losses
            )
            _sleep_table = Table(
                columns=self.sleep_table.columns, data=self.sleep_table.data
            )
            self.log({"sleep_table": _sleep_table})

            # Apply post-sleep shrink using the Fisher diagonal from the previous wake.
            if self._plasticity_decay_enabled:
                _t0 = time.time()
                self._apply_fisher_shrink()
                self._sh_time_sec["shrink"] += time.time() - _t0

        # Utility replay needs Need scores keyed by the same indices as gain
        # (the canonical candidate pool for utility selection).
        elif next_phase == "SLEEP":
            if self.sleep_mechanism_cfg.replay_criteria == "utility":
                logger.info("Computing Need scores for utility replay...")
                # Need has to be calculated at the end of wake phase because embeddings shift during training
                need_scores = compute_need_scores(
                    model=unwrap_model(self.model),
                    dataset=self.train_dataset,
                    candidate_indices=sampler.curr_wake_candidates,
                    n_layers=self.sleep_mechanism_cfg.need_embedding_layers,
                    k=self.sleep_mechanism_cfg.need_knn_k,
                    device=self.args.device,
                    batch_size=self.args.per_device_train_batch_size,
                )
                sampler.update_utility_scores(need_scores)
            # Finalize Fisher diagonal from the wake phase that just ended.
            if self._plasticity_decay_enabled:
                _t0 = time.time()
                self._finalize_fisher()
                self._sh_time_sec["score"] += time.time() - _t0

        logger.info("Switching to %s phase...", next_phase)
        sampler.switch_phase(next_phase)
        if next_phase == "SLEEP":
            logger.info("Contextualize?: %s", sampler.contextualize_sleep)
            logger.info("num candidates: %d", len(sampler.wake_candidates))
            logger.info("Replay Buffer Size: %d", len(sampler.replay_buffer))
            diag = getattr(sampler, "last_utility_diagnostics", None)
            if diag is not None:
                self.log({f"utility/{k}": v for k, v in diag.items()})

        logger.info(
            f"Swapped to {sampler.phase} phase for fold {sampler.curr_fold} of {sampler.n_phases}"
        )
        logger.info("Rebuilding train dataloader...")
        self._train_dataloader = None

        timing = {
            f"time/{phase.lower()}_phase_sec": phase_wall_sec,
            "time/train_wall_sec": (
                time.time() - self._train_wall_start
                if self._train_wall_start is not None
                else float("nan")
            ),
            "time/per_sample_grad_sec": self._sh_time_sec["per_sample_grad"],
            "time/sh_score_sec": self._sh_time_sec["score"],
            "time/sh_shrink_sec": self._sh_time_sec["shrink"],
        }
        timing["time/sh_total_sec"] = (
            timing["time/sh_score_sec"] + timing["time/sh_shrink_sec"]
        )
        self.log(timing)
        # Restart the phase clock after the eval + bookkeeping above so the
        # next phase's number is training time only.
        self._phase_wall_start = time.time()

    def _accumulate_fisher(self, grads_dict: Dict[str, torch.Tensor]) -> None:
        """Add this batch's squared per-sample gradients to the Fisher running totals."""
        # Defensive: a non-finite per-sample gradient would silently poison the
        # accumulator for the rest of the wake phase. Compute squared sums
        # first, check finiteness across all params, and skip the whole batch
        # if any are non-finite.
        squared = {}
        for name, g in grads_dict.items():
            # g: [B, *param_shape]; square and sum across the batch.
            squared[name] = (g * g).sum(dim=0).detach()
        for name, s in squared.items():
            if not torch.isfinite(s).all():
                logger.warning(
                    "Non-finite Fisher contribution at param '%s'; "
                    "skipping this batch (accumulator and sample count unchanged).",
                    name,
                )
                return

        if self.fisher_accumulator is None:
            self.fisher_accumulator = {
                name: torch.zeros(g.shape[1:], device=g.device, dtype=g.dtype)
                for name, g in grads_dict.items()
            }
        batch_size = next(iter(grads_dict.values())).shape[0]
        for name, s in squared.items():
            self.fisher_accumulator[name].add_(s)
        self.fisher_sample_count += batch_size

    def _finalize_taylor_saliency(self) -> None:
        """Score first-order shrink saliency at the end-of-wake weights.

        One per_sample_grads pass on the last wake batch, at the settled
        weights, rather than an average over the moving wake trajectory. The
        stored score is a protection priority: higher always means "protect".
        For the signed variant that means negating w*grad, because it is the
        most negative product -- where shrinking would raise the loss -- that
        deserves protection.
        """
        if self._last_wake_batch is None:
            logger.warning(
                "Taylor saliency enabled but no wake batch was captured at "
                "WAKE->SLEEP; leaving importance scores unset."
            )
            self.fisher_diagonal = None
            return

        model = unwrap_model(self.model)
        was_training = model.training
        # eval() for the measurement pass: per_sample_grads runs vmap with
        # randomness="different", so dropout would give each sample its own
        # mask and the signed products would partially cancel when summed.
        # Fisher's squared terms are immune to this; a signed sum is not.
        model.eval()
        try:
            with torch.no_grad():
                weights = {
                    name: param.detach().clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
            grads_dict = per_sample_grads(
                model,
                self._last_wake_batch["input_ids"],
                self._last_wake_batch["attention_mask"],
                self._last_wake_batch["labels"],
                self.task_name,
            )
        finally:
            if was_training:
                model.train()

        signal = self.sleep_mechanism_cfg.plasticity_decay.importance_signal
        scores = {
            name: taylor_saliency(weights[name], grad, signal)
            for name, grad in grads_dict.items()
            if name in weights
        }

        for name, score in scores.items():
            if not torch.isfinite(score).all():
                logger.warning(
                    "Non-finite Taylor saliency at param '%s'; skipping "
                    "shrink this cycle rather than protecting a garbage set.",
                    name,
                )
                self.fisher_diagonal = None
                return

        self.fisher_diagonal = scores
        self._last_wake_batch = None
        logger.info(
            "Scored %s saliency at WAKE->SLEEP over %d params.",
            signal,
            len(scores),
        )

    def _finalize_fisher(self) -> None:
        """Divide accumulated squared gradients by the sample count to get Fisher diagonal."""
        if self._taylor_signal_enabled:
            self._finalize_taylor_saliency()
            return
        if self.fisher_accumulator is None or self.fisher_sample_count == 0:
            logger.warning(
                "Plasticity decay enabled but Fisher accumulator empty at "
                "WAKE->SLEEP; skipping finalization."
            )
            # Do not leave last cycle's diagonal in place: it would be
            # reapplied at the next SLEEP->WAKE as if freshly measured.
            self.fisher_diagonal = None
            return
        count = self.fisher_sample_count
        self.fisher_diagonal = {
            name: tensor / count
            for name, tensor in self.fisher_accumulator.items()
        }
        self.fisher_accumulator = None
        self.fisher_sample_count = 0
        logger.info(
            "Finalized Fisher diagonal over %d samples across %d params.",
            count,
            len(self.fisher_diagonal),
        )

    def _apply_fisher_shrink(self) -> None:
        """Apply Fisher-protected multiplicative shrink at SLEEP->WAKE."""
        if self.fisher_diagonal is None:
            logger.warning(
                "Plasticity decay enabled but no Fisher diagonal available "
                "at SLEEP->WAKE; skipping shrink."
            )
            return
        cfg = self.sleep_mechanism_cfg.plasticity_decay
        apply_fisher_protected_shrink(
            fisher=self.fisher_diagonal,
            module=unwrap_model(self.model),
            shrink_factor=cfg.shrink_factor,
            protect_top_fraction=cfg.protect_top_fraction,
            threshold_scope=cfg.threshold_scope,
        )
        self.fisher_diagonal = None
