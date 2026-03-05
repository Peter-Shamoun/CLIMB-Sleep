<figure>
  <img src="./misc/sleep_mechanism_diagram.jpg" alt="Sleep Mechanism Diagram" style="width:100%">
  <figcaption>One phase in the Sleep Mechanism.</figcaption>
</figure>

# Sleep-Consolidated Language Modeling

### *Humans don't relive the same day ten times. Why should language models?*

Language Models (LMs), while powerful, are extremely resource-intensive to train. 
Effective LMs require hundreds of times more data to build an understanding of language than human children, often more text than a human will ever be exposed to in their entire lifetime.
Additionally, training modern language models is expensive and repetitive.
Models read the same data over and over, sometimes dozens of times, to learn effectively.
This is entirely different from how humans acquire language/knowledge: we live through every day once, and experience everything only once.
In addition to not being cognitively plausible, epoch training means that even after the model has mastered easier samples, it is still re-trained on those same samples in the same contexts as before, wasting resources and leading to overfitting.
This disconnect between the training schedules of LMs and human language acquisition could not only contribute to their inefficiency, but also means that there are strong limitations in the use and interpretation of LMs as cognitive models.

Humans, however, do revisit past experiences during sleep.
Neuroscience research has shown that sleep is not just a period of rest, but critical for developing memories.
While the waking brain is optimized for encoding new experiences into memory, during sleep, the brain undergoes a process called *memory consolidation*, where they are stabilized and integrated into pre-existing synaptic networks. 
By consolidating abstract representations of our memories in sleeping periods, humans also retain world knowledge and episodic memory of recent events (declarative memory) as well as intuition and unconscious long-term memories that influence their behavior (non-declarative memory), both of which are important for learning complex skills such as language.

It is clear that sleep is a process of utmost importance for cognitive development, and contributes to how humans are able to quickly encode and retain information from recent experiences without truly experiencing them more than once. In this project, we will explore sleep-inspired, cognitively-plausible training schedules for language models in the hopes of producing data-efficient training paradigms that diverge from standard multi-epoch conventions.


## Setup 

### Evaluation Script
BabyLM evaluation is optionally run during training, but always runs at the end of training.
Download the `evaluation_data` folder in [this OSF directory](https://osf.io/ryjfm/).
Make sure the resulting `evaluation.zip` file is stored in the root directory of this repository.

### HF Hub and WandB
In order to interact with the hub, you need to generate read and write [access tokens](https://huggingface.co/docs/hub/security-tokens) from your hugging face account. Once generated, store these values as environment variables with the names `HF_READ_TOKEN` and `HF_WRITE_TOKEN`.

Additionally, make sure you are logged in to wandb by storing your wandb API key in an environment variable called `WANDB_API_KEY`.
```
HF_READ_TOKEN = <your-read-tok>
HF_WRITE_TOKEN = <your-write-tok>
WANDB_API_KEY = <your-wandb-key>
```

### Environment Installation
Once the evaluation data is downloaded and your keys are set, run the `setup.sh` script to prepare your environment and install the evaluation pipeline.
```
./setup.sh ./evaluation_data.zip
```
If you've downloaded the eval data somewhere else, replace `./evaluation_data.zip` with the path to the data.

## Running Experiments
### Train Sleep Model

Set the appropriate hyperparameters in `conf/config.yaml`, including pointing to the correct sleep mechanism file.
You may also change the sleep hyperparameters directly in the config files within `conf/sleep_mechanism`.

For example, to train with a default set of sleep parameters, make sure in  `conf/config.yaml`, `sleep_mechanism` is set to `default`. 

Then, simply run
```
python train.py
```
This will also download and preprocess the BabyLM strict-small dataset, which is used for training.

### Run Sweep

Set your sweep ranges in the `scripts/sweep.yaml` file. Then, run 
```
python run_sweep.py
```

### Other Experiments
Various other experiments are located in different branches of this repository.
To train the corresponding model(s), switch to that branch and run `train.py`.

| **Experiment**   | **Branch**    |
|------------------|---------------|
| Gridsearch Sweep | `main`, `best_sweep_run`|
| Baselines        | `baseline_run`, `baseline_like_run` |
| Replay Experiments | `random_replay_run`, `weighted_replay_run`, `strict_replay_run`|

## Dataset

We use one of the BabyLM Challenge datasets, a curated corpus that is designed to mimic the linguistic input that children receive during early language acquisition.
Specifically, we utilize the 10M-word strict-small text-only dataset, which roughly represents the amount of word tokens a child encounters by age 13.

This dataset is made up of a combination of sources from specifically two domains:
| **Domain**              | **Source**                     | **Description**             | **Words (M)**      | **\%**        |
|-------------------------|--------------------------------|-----------------------------|--------------------|---------------|
|*Transcribed Speech*     | OpenSubtitles                  | Movie and TV subtitles      | 31.28              | 31\%          |
|*Transcribed Speech*     | QED                            | Educational video subtitles | 10.24              | 11\%          |
|*Transcribed Speech*     | British National Corpus        | Transcribed dialogue        | 8.16               | 8\%           |
|*Transcribed Speech*     | CHILDES                        | Adult-child interactions    | 4.21               | 5\%           |
|*Transcribed Speech*     | Switchboard Corpus             | Telephone conversations     | 1.18               | 1\%           |
|                         | *Subtotal*                     |                             | *55.07*            | *56\%*        |
|*Child-Directed Language*| Simple Wikipedia               | Simplified encyclopedia     | 14.66              | 15\%          |
|*Child-Directed Language*| Wikipedia                      | Standard encyclopedia       | 10.08              | 10\%          |
|*Child-Directed Language*| Children's Book Test           | Children’s books collection | 5.55               | 6\%           |
|*Child-Directed Language*| Children's Stories Text Corpus | Selected children's stories | 3.22               | 3\%           |
|*Child-Directed Language*| Standard Project Gutenberg     | Literary texts              | 9.46               | 10\%          |
|                         | *Subtotal*                     |                             | *42.97*            | *44\%*        |
|**Total**                |                                |                             | **98.04**          | **100\%**     |


This composition is meant to reflect the oral and written language input children naturally receive, with a majority coming from spoken or conversational sources to mirror how hearing children acquire language.

## Implementation Details
### Config Files
Under `/src/config.py` you will find the general structure of the hydra config file that our program expects. The purpose of explicitly defining the structure of the config in this manner is two fold 1) to show the user the set of available configurable options 2) to run type-checking on passed in configs, ensuring that the parameters and their types match this pre-defined format. 

We run automatic type-checking on all the passed in config files, and also check that there are no missing required parameters of the config file. If there are, we raise an error.

The `/conf` directory stores all the default configs and subconfigs. The entry point to the default config we use is `conf/config.yaml`. Taking a look at the `conf` directory, you will notice that each sub-directory of `conf` (i.e. `conf/data_curriculum`) stores a sub-configuration. For sleep mechanism configurations, see the `conf/sleep_mechanism` folder. There, you'll find a default config and a minimal testing config for the sleep mechanism. Choose between which of these files to use in the `conf/config.yaml` file, as the `sleep_mechanism` argument under `defaults`.

### DataLoading 

We define a custom SleepDataLoader in `/src/dataloader.py` that subclasses the normal hugging face Dataloader class. In the SleepDataLoader, unlike in the normal DataLoader, we are able to keep track of the global step number of training (i.e. how many batches of training data have already been trained on) and indices of the data we train on. This information is useful because it allows us to configure special behavior of the Trainer for different parts of training -- this is key for the functionality of the sleep data sampling. We also implement context-augmented padding within the dataloader.

We also implement the SleepSampler, in `/src/data_curriculum/sleep_sampler.py`.
This subclasses the PyTorch Sampler, and implements much of the sleep functionality, including switching between phases, limiting access to specific folds of the data, and maintaining a replay buffer.

### Preprocessing and Tokenization

Other useful methods for data preprocessing, tokenizer and inference can be found under `src/utils`.

### Evaluation

Perplexity evaluations are done within the training script and logged to Weights and Biases.
For linguistic (BabyLM) evaluations, we use the official BabyLM Evaluation Pipeline from 2025.


### Model Architecture 

For most of our experiments, we use variants of Roberta language models. The architectures and the associated configurations are specified under `/src/models`. To associate a model name with a given huggingface model and an assocaited config, we store a registry inside of the `models` package. When we load a model we query this registry. 


## References
[Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora](https://aclanthology.org/2023.conll-babylm.1/) (Warstadt et al., CoNLL-BabyLM 2023)
