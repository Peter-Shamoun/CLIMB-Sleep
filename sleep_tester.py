import logging
import os

# config-related imports
# import hydra
import torch

# training pipeline imports
from datasets import DatasetDict, load_dataset
# from hydra.core.config_store import ConfigStore
# from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
# from transformers.training_args import TrainingArguments
# from wandb.errors import CommError as WandbCommError

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

# A logger for this file
logger = logging.getLogger(__name__)

@record
# @hydra.main(version_base=None, config_path="conf", config_name="config")
def main() -> None: 
    cfg = {
        'dataset':{
            'name': 'cambridge-climb/BabyLM',
            'subconfig': 'strict_small'
        },

        'model':{
            'name': 'roberta_pre_layer_norm',
            'model_kwargs': {
                'vocab_size': 8192,
                'num_hidden_layers': 8,
                'num_attention_heads': 8, 
                'hidden_size': 256,
                'intermediate_size': 2048,
                'layer_norm_eps': 1e-5,
                'eos_token_id': 4,
                'bos_token_id': 3,
                'pad_token_id': 1,
                'tie_word_embeddings': False,
            }
        },

        'tokenizer':{
            'name': 'cambridge-climb/CamBabyTokenizer-8192',
            'add_prefix_space': True  # better if True, whether to treat first token like any other token (False in GPT-2)
        },
        
        'experiment':{
            'seed': 42 
        },

        'data_preprocessing':{
            'include_punctuation': True,
            'join_sentences': True,
            'max_input_length': 128,
        },

        'trainer':{
            'batch_size': 32, # across 4 GPUs gives an effective batch size of 128
            'lr': 1e-3, # 1e-4 is used in fairseq; 1e-3 is default in huggingface
            'num_warmup_steps': 100_000,
            'max_training_steps': 400_000,
            'eval_blimp': True,
            'eval_glue': False,
            'eval_msgs': False,
            'eval_perplexity': True
        }
    }
    
    # Setup: load dataset and create sampler
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"
    
    # Loading dataset
    logger.info("Loading dataset")
    dataset: DatasetDict = load_dataset(
        'cambridge-climb/BabyLM',
        'strict_small',
        # use_auth_token=os.environ["HF_READ_TOKEN"],
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
    
    # Create SleepSampler
    BATCH_SIZE = 32
    REPLAY_RATIO = 0.4
    N_PHASES = 5
    
    sleep_sampler = SleepSampler(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        replay_ratio=REPLAY_RATIO,
        n_phases=N_PHASES,
    )
    
    # ===Begin Tests===
    
    print(sleep_sampler.dataset)
    print(sleep_sampler.batch_size)
    print(sleep_sampler.replay_ratio)
    print(sleep_sampler.n_phases)
    
    print(sleep_sampler.dataset_indices)
    print(sleep_sampler.folds)

if __name__ == "__main__":
    main()