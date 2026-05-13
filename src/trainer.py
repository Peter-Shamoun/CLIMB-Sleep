""" Main trainer class for BabyLM. """

import copy
import logging
import os
import shutil
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

import torch
from torch import Tensor, device
import torch.distributed as dist
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from huggingface_hub.hf_api import create_repo
# from huggingface_hub.repository import Repository
# from huggingface_hub.utils._errors import HfHubHTTPError
from omegaconf import OmegaConf

# Data loading
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Model Training
from transformers import (
    PreTrainedTokenizerFast, 
    Trainer, 
    TrainerCallback,
    DataCollatorForLanguageModeling,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)
from transformers.utils import is_torch_neuroncore_available
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import (
    HubStrategy,
    IntervalStrategy,
    has_length,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
# from transformers.utils import (
#     get_full_repo_name,
#     is_torch_neuroncore_available,
# )
from transformers.models.roberta_prelayernorm.modeling_roberta_prelayernorm import (
    RobertaPreLayerNormLMHead,
)

# Model Loading
from src.models import load_base_model
from src.utils.data import base_collate_fn
from src.utils.inference import (
    compute_trainer_perplexity,
)
from src.data_curriculum.contextualize_collate import context_augmented_collate
from wandb import Table

# typing imports
from .config import BabyLMConfig

# Sleep Sampling
from .data_curriculum.sleep_sampler import SleepSampler

# Sleep Data Loader 
from .dataloader import SleepDataLoader

# Model Evaluation
from .evaluator import BlimpEvaluator, FinetuneEvaluator, BabyLMEvaluator

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    def __init__(
        self,
        hydra_config: BabyLMConfig,
        dry_run: bool,
        args: TrainingArguments,
        tokenizer: PreTrainedTokenizerFast,
        sleep_table: Table,
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

        super().__init__(args=args, **kwargs)

        self.sleep_mechanism_cfg = hydra_config.sleep_mechanism
        if self.sleep_mechanism_cfg:
            total_wake_steps = min(
                        self.sleep_mechanism_cfg.wake_block_steps * self.sleep_mechanism_cfg.n_phases,
                        len(self.train_dataset) // self.args.per_device_train_batch_size
                        )
            self.sleep_max_steps = (
                (args.max_steps
                    - total_wake_steps)
                // self.sleep_mechanism_cfg.n_phases
            )
            if self.sleep_max_steps < 1:
                min_steps = (total_wake_steps
                             + self.sleep_mechanism_cfg.n_phases)
                logger.info("Too few steps for training! Min steps: %d" % min_steps)
                raise ValueError("Too few steps for training! Min steps: %d" % min_steps)
        logger.info("Wake steps/phase: %d" % math.ceil(total_wake_steps / self.sleep_mechanism_cfg.n_phases))
        logger.info("Sleep steps/phase: %d" % self.sleep_max_steps)
        self.phase_steps = 0

        # NOTE: The hidden dimension of the base model (is the input dimension to the task head)
        # We check that this variable is set in the config file when loading the base model

        self.hidden_rep_size = hydra_config.model.model_kwargs["hidden_size"]
        if self.sleep_mechanism_cfg:
            logger.info(
                f"Using sleep mechanism configuration {self.sleep_mechanism_cfg}"
            )

        self.tokenizer = tokenizer
        
        # track gradient steps for checkpointing
        self.global_step = 0
        
        # Initialize MLM task head
        mlm_head_config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size,  # type: ignore
            hidden_size=self.hidden_rep_size
        )

        self.mlm_head = RobertaPreLayerNormLMHead(mlm_head_config).to(
            self.args.device
        )
        
        # Setting up optimizer and scheduler for task head
        self.mlm_optimizer = AdamW(
            self.mlm_head.parameters(),
            lr=1e-3,
        )

        self.mlm_scheduler = get_linear_schedule_with_warmup(
            self.mlm_optimizer,
            num_warmup_steps=args.max_steps // 10,
            num_training_steps=args.max_steps,
        )

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
                logger.info("Using SleepSampler for sleep-consolidated learning")

                self._train_sampler = SleepSampler(
                    dataset=self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    replay_ratio=self.sleep_mechanism_cfg.replay_ratio,
                    n_phases=self.sleep_mechanism_cfg.n_phases,
                    max_seq_length=self.sleep_mechanism_cfg.max_seq_length,
                    contextualize_sleep=True,
                    n_augmentations=self.sleep_mechanism_cfg.n_augmentations,
                    replay_strategy=self.sleep_mechanism_cfg.replay_strategy
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
        logger.info(f"GLOBAL STEP: {self.state.global_step};{self.global_step}")

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
        if self.sleep_mechanism_cfg and isinstance(train_sampler, SleepSampler):
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
                pin_memory=self.args.dataloader_pin_memory
            )

        assert (
            self.tokenizer is not None
        ), "Tokenizer is not set. Please set the tokenizer before calling the train method."

    def compute_loss(self, 
                     model, 
                     inputs, 
                     override_input_ids=None,
                     override_labels=None,
                     loss_kwargs=None,
                     inference=False,
                     **kwargs):
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
            and hasattr(self, 'callback_handler')
            and hasattr(self.callback_handler, 'train_dataloader')
            and self.callback_handler.train_dataloader is not None
            and hasattr(self.callback_handler.train_dataloader, 'sampler')
            and hasattr(self.callback_handler.train_dataloader.sampler, 'phase')
            and self.callback_handler.train_dataloader.sampler.phase == "WAKE"
        )
        input_ids = (
            override_input_ids
            if override_input_ids is not None
            else inputs["input_ids"]
        )
        base_model_outputs = model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"]
            if "attention_mask" in inputs
            else None,
        )
        base_model_hidden_states = base_model_outputs[0]
        
        logits = self.mlm_head(base_model_hidden_states).transpose(-1, -2)
        labels = (
            override_labels
            if override_labels is not None
            else inputs["labels"]
        )
        
        # Compute the loss
        if track_per_sample:
            # Compute per-sample loss for sleep mechanism replay buffer
            # logger.info("Computing per-sample loss")
            per_sample_loss = cross_entropy(logits, labels, reduction='none', **(loss_kwargs or {}))
            per_sample_loss = per_sample_loss.mean(dim=-1) # avgs across samples
            # loss = per_sample_loss.mean()
            # Add per-sample loss to sampler
            if "indices" in inputs:
                indices = inputs['indices'].tolist()
                losses = per_sample_loss.detach().cpu().tolist()
                self.callback_handler.train_dataloader.sampler.add_to_candidates(indices, losses)
        # else:
            # logger.info("Computing loss")
        loss = cross_entropy(logits, labels, **(loss_kwargs or {}))
        if inference: # Return loss early if inference
            return loss
        # averaging over the processes
        total_unit_loss_scalar = self._nested_gather(loss).mean().item()  # type: ignore
        loss_metrics["loss_mlm"] = total_unit_loss_scalar
        if self.sleep_mechanism_cfg:
            curr_phase = self.callback_handler.train_dataloader.sampler.phase
            loss_metrics[f"loss_mlm_{curr_phase}"] = total_unit_loss_scalar
            
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
        logger.info(f"LOSS: {loss}")
        return loss
    
    def training_step(self, model, inputs, *args):
        loss = super().training_step(model, inputs)

        # == SLEEP MECHANISM == #
        if self.sleep_mechanism_cfg:
            # if wake phase over, reset phase steps, eval, then switch
            sampler = self.callback_handler.train_dataloader.sampler
            phase = sampler.phase
            phase_max = (min(self.sleep_mechanism_cfg.wake_block_steps, sampler.wake_max_steps) 
                         if phase == 'WAKE' 
                         else self.sleep_max_steps)

            if self.phase_steps >= phase_max:
                if (phase == "WAKE"
                and (self.sleep_mechanism_cfg.wake_block_steps 
                     > sampler.wake_max_steps)):
                    logger.info("WARNING: The selected WAKE phase max steps exceeds the size of the current fold. Ending WAKE phase early.")
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
                evaluator_metrics[
                    f"{metric_key_prefix}_{key}"
                ] = evaluator_metrics.pop(key)

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
        Initialize a full language model that includes the base model and the mlm head.
        """

        # copy hydra config and change base_model to include mlm head
        lm_config = copy.deepcopy(self.hydra_config)
        lm_config.model.name = lm_config.model.name + "_mlm"

        lm_model = load_base_model(lm_config)

        # unwrapping the base model and the mlm task head and copying that over into the lm model
        setattr(
            lm_model,
            f"{lm_model.base_model_prefix}",
            unwrap_model(self.model.base_model),
        )
        lm_model.lm_head = unwrap_model(
            self.mlm_head
        )

        return lm_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Override the Trainer._save() method to save the objective curriculum state as well,
        and to save the full language model (base model + mlm head).
        """

        if self.args.should_save:
            super()._save(output_dir=output_dir, state_dict=state_dict)

            # Saving should be done only on the main process

            # NOTE: We need to save the objective mlm head state as well
            output_dir = (
                output_dir if output_dir is not None else self.args.output_dir
            )

            # save step count
            step_file = os.path.join(output_dir, "trainer_state.json")
            # write current step count to file inside checkpoint folder
            with open(step_file, 'w') as f:
                json.dump({'global_step': self.global_step}, f)

            mlm_model_dir = os.path.join(output_dir, "lm_model")
            task_heads_dir = os.path.join(output_dir, "task_heads")
            os.makedirs(mlm_model_dir, exist_ok=True)
            os.makedirs(task_heads_dir, exist_ok=True)

            # save the full language model + the associated tokenizer (for inference)
            lm_model = self._initialize_full_lm_model()
            lm_model.save_pretrained(mlm_model_dir)

            self.tokenizer.save_pretrained(mlm_model_dir)

            torch.save(
                unwrap_model(self.mlm_head).state_dict(),
                os.path.join(task_heads_dir, f"mlm_task_head.pt"),
            )

            torch.save(
                self.mlm_optimizer.state_dict(),
                os.path.join(task_heads_dir, f"mlm_optimizer.pt"),
            )
            torch.save(
                self.mlm_scheduler.state_dict(),
                os.path.join(task_heads_dir, f"mlm_scheduler.pt"),
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        task_head_dir = os.path.join(resume_from_checkpoint, "task_heads")
        self.mlm_head.load_state_dict(
            torch.load(
                os.path.join(
                    task_head_dir, f"{self.task_unit_name}_task_head.pt"
                ),
                map_location=self.args.device,
            )
        )
        self.mlm_optimizer.load_state_dict(
            torch.load(
                os.path.join(task_head_dir, f"{self.task_unit_name}_optimizer.pt"),
                map_location=self.args.device,
            )
        )
        self.mlm_scheduler.load_state_dict(
            torch.load(
                os.path.join(task_head_dir, f"{self.task_unit_name}_scheduler.pt"),
                map_location=self.args.device,
            )
        )

        # load step count
        step_file = os.path.join(resume_from_checkpoint, "trainer_state.json")
        # reads from JSON file and restores step count
        if os.path.exists(step_file):
            with open(step_file, 'r') as f:
                state = json.load(f)
                self.global_step = state.get('global_step', 0)


    def _load_best_model(self):
        super()._load_best_model()

        task_head_dir = os.path.join(
            self.state.best_model_checkpoint, "task_heads"
        )
        self.mlm_head.load_state_dict(
            torch.load(
                os.path.join(
                    task_head_dir, f"mlm_task_head.pt"
                ),
                map_location=self.args.device,
            )
        )
        self.mlm_optimizer.load_state_dict(
            torch.load(
                os.path.join(task_head_dir, f"mlm_optimizer.pt"),
                map_location=self.args.device,
            )
        )
        self.mlm_scheduler.load_state_dict(
            torch.load(
                os.path.join(task_head_dir, f"mlm_scheduler.pt"),
                map_location=self.args.device,
            )
        )

    def _wrap_model(self, model):
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs[
                    "find_unused_parameters"
                ] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs[
                    "find_unused_parameters"
                ] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            if is_torch_neuroncore_available():
                return model
            if any(p.requires_grad for p in model.parameters()):
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.args.local_rank]
                    if self.args._n_gpu != 0
                    else None,
                    output_device=self.args.local_rank
                    if self.args._n_gpu != 0
                    else None,
                    broadcast_buffers=False,
                    **kwargs,
                )

        return model
        
    def train(self, *args, resume_from_checkpoint=None, **kwargs):
        """
        Override train to re-intialize phase steps.
        """
        self.phase_steps = 0
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs) #CHANGED from super().super().train to super().train
    
    def _swap_phase(self, 
                    sampler: SleepSampler,
                    phase:str, 
                    next_phase:str):
        logger.info("Ending %s phase...", phase)
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
                [f"{i}: {self.tokenizer.decode(ids)}" for i, ids in enumerate(sample_ids)]
            )
            
            self.sleep_table.add_data(
                self.global_step, # global_step
                self.phase_steps, # phase_step
                sampler.curr_fold, # phase_num
                sample_texts, # replay_samples
                # losses # losses
            )
            _sleep_table = Table(
                columns=self.sleep_table.columns,
                data=self.sleep_table.data
            )
            self.log({
                "sleep_table": _sleep_table
            })
        
        logger.info("Switching to %s phase...", next_phase)
        sampler.switch_phase(next_phase)
        if next_phase == "SLEEP":
            logger.info("Contextualize?: %s", sampler.contextualize_sleep)
            logger.info("num candidates: %d", len(sampler.wake_candidates))
            logger.info("Replay Buffer Size: %d", len(sampler.replay_buffer))
            
        logger.info(f"Swapped to {sampler.phase} phase for fold {sampler.curr_fold} of {sampler.n_phases}")
        logger.info("Rebuilding train dataloader...")
        self._train_dataloader = None