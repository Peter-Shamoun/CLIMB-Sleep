"""Defines the set of hyperparameters to be specified in the config file."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from omegaconf import MISSING, DictConfig
from enum import Enum


@dataclass
class ExperimentParams(DictConfig):
    seed: int

    # Name of the experiment - needs to be set at runtime
    name: str = MISSING

    # Name of the group that the current experiment belongs to
    # analogous to 'project' in wandb
    group: str = MISSING

    # whether to run a minimal version of the experiment
    dry_run: bool = False

    # whether to run the experiment only offline
    offline_run: bool = False

    # Optional checkpoint path to resume training from
    resume_checkpoint_path: Optional[str] = None

    # If resume_checkpoint_path is not None and we are logging to wandb,
    # we need to specify the run_id of the run we are resuming from
    resume_run_id: Optional[str] = None


@dataclass
class DatasetParams(DictConfig):
    # name of the dataset on huggingface
    name: str
    # subconfig i.e. strict-small
    subconfig: str


@dataclass
class TokenizerParams(DictConfig):
    # data processing parameters
    name: str

    # additional optional kwargs
    add_prefix_space: Optional[bool] = None


@dataclass
class DataPreprocessingParams(DictConfig):
    # params for preprocessing the dataset (i.e. tokenization)
    include_punctuation: bool
    join_sentences: bool
    max_input_length: int
    callback_functions: Optional[List[str]] = None


@dataclass
class ModelParams(DictConfig):
    # model parameters
    name: str

    # NOTE: At least 'hidden_size' needs to be specified
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerParams(DictConfig):
    batch_size: int
    lr: float
    num_warmup_steps: int
    max_training_steps: int
    eval_blimp: bool
    eval_glue: bool
    eval_msgs: bool
    eval_perplexity: bool


### Curriculum learning parameter: can be either objective or data-driven ###


## Objective curriculum learning parameters ##
@dataclass
class ObjectiveCurriculumUnitParams(DictConfig):

    # parameters for the task head architecture
    task_head_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    # parameters for the optimizer
    optimizer_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    # parameters for the scheduler
    scheduler_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Additional optional kwargs dependent on the objective curriculum unit
    optional_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ObjectiveCurriculumParams(DictConfig):
    # objective curriculum learning parameters

    units: Dict[str, ObjectiveCurriculumUnitParams]
    steps: Dict[str, List[float]]


## Data-driven curriculum learning parameters ##
@dataclass
class PacingFunctionParams(Mapping[str, Any]):
    # Num of steps to take (in percent) before beginning the curriculum
    start_percent: float
    # Num of steps to take (in percent) before ending the curriculum
    end_percent: float
    # Difficulty (percentile of the data) to start at
    starting_difficulty: float
    # Max difficulty (percentile of the data) to end at; 1.0 means include all data at the
    # end of the curriculum
    max_difficulty: Optional[float] = 1.0


# Difficulty Scorer Parameters

DifficultyScorerKwargsType = Optional[Dict[str, Any]]


@dataclass
class DataCurriculumParams(DictConfig):
    # data-driven curriculum learning parameters

    # the column of the data to sort by (aka n_gram perplexity, sentence length, etc.)
    difficulty_scorer_name: str

    difficulty_scorer_kwargs: DifficultyScorerKwargsType

    # one of ['linear', 'quad', 'root', 'step', 'exp', 'log'] or None, meaning no pacing
    pacing_fn_name: str

    pacing_fn_kwargs: PacingFunctionParams


## Vocabulary curriculum parameters ##
@dataclass
class VocabularyCurriculumParams(DictConfig):
    # vocabulary curriculum learning parameters

    # the curriculum used to determine which tokens to map to <unk> (aka token_ids, part of speech etc.)
    vocabulary_curriculum_name: str

    # one of ['linear', 'quad', 'root', 'step', 'exp', 'log'] or None, meaning no pacing
    pacing_fn_name: str

    pacing_fn_kwargs: PacingFunctionParams

# Plasticity decay for sleep mechanism
class PlasticityDecayType(str, Enum):
    lr_decay = "lr_decay"
    freeze_layers = "freeze_layers"
    pruning = "pruning"


# Sleep mechanism params
@dataclass
class SleepMechanismParams(DictConfig):
    # Number of steps to train on new data before entering sleep
    wake_block_steps: int
    # Maximum number of steps allowed in a sleep phase
    sleep_max_steps: int
    # Target loss value to exit sleep phase early
    # SHOULD WE MAKE THIS OPTIONAL??
    # sleep_loss_threshold: Optional[float] = None
    sleep_loss_threshold: float
    # Percentage/Fraction/Ratio of high-loss samples to keep (0.1 for top 10%)
    replay_ratio: float = 0.1
    # Plasticity decay type
    plasticity_decay_type: PlasticityDecayType = PlasticityDecayType.lr_decay
    # Factor by which to decay plasticity
    plasticity_decay_rate: float = 0.9


### Container for entire config ###


@dataclass
class BabyLMConfig(DictConfig):
    experiment: ExperimentParams
    dataset: DatasetParams
    tokenizer: TokenizerParams
    data_preprocessing: DataPreprocessingParams
    model: ModelParams
    trainer: TrainerParams
    objective_curriculum: ObjectiveCurriculumParams
    data_curriculum: Optional[DataCurriculumParams] = None
    vocabulary_curriculum: Optional[VocabularyCurriculumParams] = None

    sleep_mechanism: Optional[SleepMechanismParams] = None