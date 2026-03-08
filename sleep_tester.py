import logging
import os

# config-related imports
import hydra
import torch
import numpy as np

# training pipeline imports
from datasets import DatasetDict, load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from transformers.training_args import TrainingArguments
from wandb.errors import CommError as WandbCommError

# from hydra.core.config_store import ConfigStore
# from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from transformers.training_args import TrainingArguments
from wandb.errors import CommError as WandbCommError

# wandb for logging metrics
# import wandb
from src.config import BabyLMConfig
from src.evaluator import collect_results
from src.models import load_base_model
from src.tokenizer import load_tokenizer
from src.trainer import CustomTrainer
from src.utils.data import DatasetPreprocessor
from src.utils.setup import set_seed
from src.data_curriculum.sleep_sampler import SleepSampler

cs = ConfigStore.instance()
cs.store(name="base_config", node=BabyLMConfig)

# A logger for this file
logger = logging.getLogger(__name__)

@record
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: BabyLMConfig) -> None: 
    
    # Setup: load dataset and create sampler
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"
    
    logger.info("Config: %s", OmegaConf.to_yaml(cfg))
    
    # Loading dataset
    logger.info("Loading dataset")
    dataset: DatasetDict = load_dataset(
        'cambridge-climb/BabyLM',
        'strict_small',
        # use_auth_token=os.environ["HF_READ_TOKEN"],
        trust_remote_code=True,
    )  # type: ignore
    
    assert isinstance(dataset, DatasetDict), "Dataset is not a DatasetDict"
    
    # Setup: load tokenizer and models
    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(cfg)

    # logger.info("Initializing model")
    # model = load_base_model(cfg)

    # assert (
    #     tokenizer.vocab_size == model.config.vocab_size
    # ), "Tokenizer and model vocab size mismatch"

    # Preprocess data
    logger.info("Preprocessing data")
    data_preprocessor = DatasetPreprocessor(cfg, tokenizer)
    
    train_dataset = dataset["train"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["train"].column_names,
    )
    
    sleep_sampler = SleepSampler(
        dataset=train_dataset,
        batch_size=cfg.trainer.batch_size,
        replay_ratio=cfg.sleep_mechanism.replay_ratio,
        n_phases=cfg.sleep_mechanism.n_phases,
    )
    
    # ===Begin Tests===
    
    print("Dataset:", sleep_sampler.dataset)
    print("===Hyperparameters===")
    print("Batch size:",sleep_sampler.batch_size)
    print("Replay ratio:", sleep_sampler.replay_ratio)
    print("N phases:", sleep_sampler.n_phases)
    print("N augmentations:", sleep_sampler.n_augmentations)
    print("Max seq length:", sleep_sampler.max_seq_length)
    print("Contextualize sleep:", sleep_sampler.contextualize_sleep)
    print("Dataset folds:")
    for i, fold in enumerate(sleep_sampler.folds):
        print(f"    len(fold {i}): {len(fold)}")
    print()
    
    print("===Sampling===")
    print("=Single Index=")
    sleep_iter = iter(sleep_sampler)
    idx = next(sleep_iter)
    print("Sampled index:", idx)
    print("Sample:")
    print(sleep_sampler.dataset[idx])
    print("Adding to candidates:")
    print("=Batch=")
    batch = []
    for i in range(sleep_sampler.batch_size):
        batch.append(next(sleep_iter))
    print("Sampled batch size:", len(batch))
    print("First index in batch:", batch[0])
    print("First sample in batch:", sleep_sampler.dataset[batch[0]])
    losses = np.random.rand(len(batch)).tolist()
    sleep_sampler.add_to_candidates(batch, losses)
    print("Indices:", batch)
    print("Dummy losses:", losses)
    print("Current wake candidates:", sleep_sampler.wake_candidates)
    print()
    
    print("===Adding to Replay Buffer===")
    print()
    print("Loss replay strategy")
    sleep_sampler.replay_strategy = "loss"
    sleep_sampler.update_replay_buffer()
    print("Replay buffer size:", len(sleep_sampler.replay_buffer))
    print("Replay buffer contents:", sleep_sampler.replay_buffer[:10])  # Show first 10 elements
    print("Replay Buffer avg. loss:", np.mean([sleep_sampler.wake_candidates[i] for i in sleep_sampler.replay_buffer]))
    print()
    print("Random replay strategy")
    sleep_sampler.replay_strategy = "random"
    sleep_sampler.update_replay_buffer()
    print("Replay buffer size:", len(sleep_sampler.replay_buffer))
    print("Replay buffer contents:", sleep_sampler.replay_buffer[:10])
    print("Replay Buffer avg. loss:", np.mean([sleep_sampler.wake_candidates[i] for i in sleep_sampler.replay_buffer]))
    print()
    print("Loss-weighted replay strategy")
    sleep_sampler.replay_strategy = "loss_weighted"
    sleep_sampler.update_replay_buffer()
    print("Replay buffer size:", len(sleep_sampler.replay_buffer))
    print("Replay buffer contents:", sleep_sampler.replay_buffer[:10])
    print("Replay Buffer avg. loss:", np.mean([sleep_sampler.wake_candidates[i] for i in sleep_sampler.replay_buffer]))
    print()
    
    print("===Phase Transition===")
    print("Current phase:", sleep_sampler.phase)
    sleep_sampler.switch_phase("SLEEP")
    print("Switched to phase:", sleep_sampler.phase)
    
    
    print("===Wake Phase===")
    

if __name__ == "__main__":
    main()