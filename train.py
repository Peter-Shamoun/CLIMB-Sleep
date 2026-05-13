"""Train a RoBERTa model on the BabyLM dataset."""

import logging
import os
import argparse
import math

# config-related imports
import hydra
import torch

# training pipeline imports
from datasets import DatasetDict, load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from transformers.training_args import TrainingArguments
from wandb.errors import CommError as WandbCommError

# wandb for logging metrics
import wandb
from src.config import BabyLMConfig
from src.evaluator import collect_results
from src.models import load_base_model
from src.tokenizer import load_tokenizer
from src.trainer import CustomTrainer
from src.utils.data import DatasetPreprocessor
from src.utils.setup import set_seed

# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=BabyLMConfig)

# A logger for this file
logger = logging.getLogger(__name__)

@record
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: BabyLMConfig):
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing keys in config: \n {missing_keys}")

    logger.info("Config: %s", OmegaConf.to_yaml(cfg))

    if cfg.sleep_mechanism:
        logger.info("Sleep mechanism enabled with %s phases", cfg.sleep_mechanism.n_phases)

    # Set seed
    set_seed(cfg.experiment.seed)

    # Loading dataset
    logger.info("Loading dataset")
    dataset: DatasetDict = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        # use_auth_token=os.environ["HF_READ_TOKEN"],
        trust_remote_code=True
    )  # type: ignore

    assert isinstance(dataset, DatasetDict), "Dataset is not a DatasetDict"

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(cfg)

    logger.info("Initializing model")
    model = load_base_model(cfg)

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        logger.info("Model and Tokenizer Mismatch - resizing model token embeddings")
        model.resize_token_embeddings(len(tokenizer))

    # Preprocess data
    logger.info("Preprocessing data")
    data_preprocessor = DatasetPreprocessor(cfg, tokenizer)

    train_dataset = dataset["train"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["train"].column_names,
    )

    eval_dataset = dataset["validation"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["validation"].column_names,
    )

    # Setting up wandb
    if cfg.experiment.offline_run:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        sleep_table = None
    else:
        # These environment variables get picked up by Trainer
        os.environ["WANDB_PROJECT"] = cfg.experiment.group
        os.environ["WANDB_ENTITY"] = cfg.experiment.entity
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        if cfg.experiment.resume_checkpoint_path:
            resume_run_id = cfg.experiment.resume_run_id
            if resume_run_id is None:
                raise RuntimeError(
                    "resume_run_id must be set if resume_checkpoint_path is set"
                )
            os.environ["WANDB_RUN_ID"] = resume_run_id
            os.environ["WANDB_RESUME"] = "allow"

        # Check if we're on process 0
        if int(os.environ.get("RANK", "0")) == 0:
            wandb.init(
                entity=cfg.experiment.entity,
                project=cfg.experiment.group,
                name=cfg.experiment.name,
                config=wandb.config,  # type: ignore
                id=cfg.experiment.resume_run_id,
                resume="allow",
            )
            if cfg.sleep_mechanism:
                sleep_table = wandb.Table(
                    columns=[
                        "global_step",
                        "phase_step",
                        "phase_num",
                        "replay_samples",
                        # "losses",
                    ]
                )
            else:
                sleep_table = None

    # Set up training arguments
    # set up max steps
    max_training_steps = cfg.trainer.max_training_steps
    steps_kwargs = {}
    if cfg.sleep_mechanism:
        total_wake_steps = min(
            cfg.sleep_mechanism.wake_block_steps * cfg.sleep_mechanism.n_phases,
            math.ceil(len(train_dataset) / cfg.trainer.batch_size)
        )
        wake_steps_per_phase = math.ceil(total_wake_steps / cfg.sleep_mechanism.n_phases)
        
        # Set according to ratio
        if cfg.sleep_mechanism.sleep_wake_ratio > 0:
            sleep_max_steps_per_phase = (
                wake_steps_per_phase
                * cfg.sleep_mechanism.sleep_wake_ratio
            )
            total_sleep_steps = sleep_max_steps_per_phase * cfg.sleep_mechanism.n_phases
            max_training_steps = total_wake_steps + total_sleep_steps
            
        # Set according to max steps
        else:
            total_sleep_steps = max_training_steps - total_wake_steps
            sleep_max_steps_per_phase = (
                math.ceil(total_sleep_steps / cfg.sleep_mechanism.n_phases)
            )
            
        if sleep_max_steps_per_phase < 1:
            min_steps = (total_wake_steps
                        + cfg.sleep_mechanism.n_phases)
            logger.info("Too few steps for training! Min steps: %d" % min_steps)
            raise ValueError("Too few steps for training! Min steps: %d" % min_steps)
        logger.info("Wake steps/phase: %d" % wake_steps_per_phase)
        logger.info("Sleep steps/phase: %d" % sleep_max_steps_per_phase)
        logger.info("Sleep/Wake ratio: %f" % (sleep_max_steps_per_phase / wake_steps_per_phase))
        logger.info("Total max steps: %d" % max_training_steps)
        steps_kwargs = {
            "sleep_max_steps_per_phase": sleep_max_steps_per_phase,
            "wake_steps_per_phase": wake_steps_per_phase
        }
        
    logging_steps = (max_training_steps 
                     // (100 if cfg.experiment.dry_run else 1000))
    logging_steps = logging_steps if logging_steps > 1 else max_training_steps
    training_args = TrainingArguments(
        output_dir=f"{cfg.experiment.output_dir}/checkpoints/{cfg.experiment.group}/{cfg.experiment.name}",
        # overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=False,
        per_device_train_batch_size=cfg.trainer.batch_size,  # NOTE: We can should maybe use auto_find_batch_size
        learning_rate=cfg.trainer.lr,
        max_steps=max_training_steps,
        warmup_steps=cfg.trainer.num_warmup_steps,
        seed=cfg.experiment.seed,
        eval_strategy="steps",
        eval_steps=max_training_steps
        // (2 if cfg.experiment.dry_run else 8),  # eval every 25% of training
        save_steps=max_training_steps
        // (
            2 if cfg.experiment.dry_run else 8
        ),  # checkpoint every 25% of training
        logging_steps=logging_steps,
        run_name=cfg.experiment.name,
        report_to=["wandb"]
        if not cfg.experiment.offline_run
        else None,  # wandb deactivated for offline runs
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=False,
        hub_model_id=None,
        hub_token=os.environ["HF_WRITE_TOKEN"],
        dataloader_drop_last=(cfg.data_curriculum is not None or cfg.sleep_mechanism is not None),
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_perplexity_mean",
        greater_is_better=False,  # smaller perplexity is better
        ddp_find_unused_parameters=False,
        ddp_timeout=28800,  # 8 hours (default is 30 minutes)
    )

    # Set up trainer
    trainer = CustomTrainer(
        hydra_config=cfg,
        dry_run=cfg.experiment.dry_run,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        sleep_table=sleep_table,
        **steps_kwargs
        # callbacks=[SleepCallback(cfg.sleep_mechanism.n_phases)],
    )
    if not cfg.experiment.resume_checkpoint_path:
        trainer.evaluate()  # Initial model evaluation
    trainer.train(resume_from_checkpoint=cfg.experiment.resume_checkpoint_path)

    logger.info("Training complete!")
    
    # Always do fast eval at the end
    trainer.eval_fast = True
    trainer.evaluate(
        metric_key_prefix="eval_best"
    )  # Note that this will also save the best model in the main output directory
    collect_results(os.path.join(trainer.args.output_dir, "lm_model"))

    trainer.save_model(
        output_dir=os.path.join(training_args.output_dir, "best_model")
    )


if __name__ == "__main__":
    main()
