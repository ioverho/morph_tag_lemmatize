<h1 align="center">
  Morphological Tagging and Lemmatization in Context
</h1>

<h4 align="center">MSc AI Thesis @ University of Amsterdam</h4>

<p align="center">
<a href="https://www.notion.so/MSc-AI-Thesis-9c3ba8027f6b4e3a82f0e391a6db76a9">
    <img
    src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white"
    alt="This project's Notion page."
    style="float: center;"
    />
</a>
<a href="https://wandb.ai/verhivo/morphological_tagging_v2?workspace=user-verhivo">
    <img src="https://img.shields.io/badge/WandB-%23000000.svg?&style=for-the-badge&logo=weightsandbiases&logoColor=#FFBE00"
    alt="This project's WandB page."
    style="float: center;"
    />
</a>
</p>

- [Models](#models)
  - [UDPipe2](#udpipe2)
    - [Differences](#differences)
    - [Training](#training)
    - [Evaluating](#evaluating)
    - [Pipeline Creation/Loading](#pipeline-creationloading)
- [References](#references)

# Models
All pre-trained models in pipeline format (including dictionaries to allow mapping from raw text to input and output to processed text) can be found (TODO: add link).

An early design choice was to opt for seq-first for UDPipe2, but batch-first for others. This make dataset files incompatible between models, unfortunately.

Generally, for deep learning experiments no reproduction is exact, and this project is no exception. Differences are detailed for each model in their respective section, ordered by expected impact (largest to smallest).
## UDPipe2
<p align="center">
    <img
    src="./misc/figures/UDPipe2 Pipeline.png"
    alt="UDPipe2's pipeline"
    style="float: center;"
    />
</p>

Designed and implemented by M. Straka and colleagues at UFAL Prague, the second version of the UDPipe pipeline has shown excellent performance in several competitions. It consists of an entirely modular system, processing various forms of pre-trained and trainable word embeddings via a residual RNN, before classifying a tokens' morphological tag set and lemma edit script.

Ultimately, this UDPipe2 was 1 of 3 winners at CONLL 2018's shared task, and 1 of 2 at CONLL/SIGMORPHON 2019 shared task 2, and the winner at the EvaLatin 2020 shared task.

Relative to other models in this repository, the backward step is small. This results in relatively fast training, even for the largest datasets. Due to the emphasis on combining pre-trained word/context-embeddings, performance is strong almost out-of-the-box.

For more information, see the [dedicated website](https://ufal.mff.cuni.cz/udpipe/2), the [Github repository](https://github.com/ufal/udpipe/tree/udpipe-2), or any of the published system's papers[^UDPipe2 CONLL][^UDPipe2 SIGMORPHON][^UDPipe2 EvaLatin].


### Differences
1. **Morph. tag factoring**: due to some morphological tags not being present in the initial UniMorph schema, and lack of detail regarding implementation, regularisation was not conducted via factoring the tags into their classes. Rather, the model was further tasked with seperately predicting presence of a cateogry
2. **Sparse embeddings**: PyTorch's sparse word embeddings and LazyAdam resulted in some very nasty optimization errors. Instead, non-sparse variants are used. This proved equally fast, and likely provided some additional reguralization
3. **Additional Reguralization**: overfit seems the most prevalent issue. As such, some additional regularization methods were applied. Both tokens and characters are masked (with low likelihood) prior to being fed into their respective models. Where possible, weight-decay was applied via AdamW

### Training
To train UDPipe2 from scratch, run:
```bash
python -u train_tagger.py --config-name udpipe2_experiment hydra/job_logging=disabled hydra/hydra_logging=disabled
```
Hydra's override syntax can be used to alter practically any aspect of training.

<details>
<summary>Arguments</summary>
<p>The [default config file](./config/udpipe2_experiment) post-processing looks like:

```yaml
# From ./config/default_train.yaml
# Experiment setup default for all models
print_hparams: False
prog_bar_refresh_rate: 200

monitor: valid/clf_agg
monitor_mode: "max"
save_checkpoints: True
save_top_k: 1

seed: 610
gpu: 1
deterministic: False
debug: False
fdev_run: False

logging:
  logger: wandb
  logger_kwargs:
    project: morphological_tagging_v2
    log_model: True
    offline: False

# From ./config/udpipe2_experiment.yaml
# Experiment setup specific to UDPipe2
experiment_name: UDPipe2
architecture: udpipe2

data:
  language: English
  treebank_name: ATIS
  batch_first: False
  len_sorted: True
  batch_size: 32
  source: ./morphological_tagging/data/um-treebanks-v2.9

trainer:
  gradient_clip_val: 2
  max_epochs: 60
  num_sanity_val_steps: 0

# From ./config/preprocessor/udpipe2.yaml
# These get fed to the UDPipe2 model, then to the UDPipe2Preprocessor class
preprocessor:
    word_embeddings: True
    context_embeddings: True
    tokenizer: None
    language: English
    lower_case_backup: False
    transformer_name: bert-base-multilingual-cased
    transformer_dropout: null
    layer_pooling: average
    n_layers_pooling: 4
    wordpiece_pooling: first

# From ./config/model/udpipe2.yaml
# These get fed to the UDPipe2 model
model:
    c2w_kwargs:
        embedding_dim: 256
        h_dim: 256
        out_dim: 256
        bidirectional: True
        rnn_type: gru
        batch_first: False
        dropout: 0.5
    w_embedding_dim: 512
    word_rnn_kwargs:
        h_dim: 512
        bidirectional: True
        rnn_type: lstm
        num_layers: 3
        residual: True
        batch_first: False
        dropout: 0.5
    char_mask_p: 0.1
    token_mask_p: 0.2
    label_smoothing: 0.03
    reg_loss_weight: 2
    lr: 1.0e-3
    betas:
    - 0.9
    - 0.99
    weight_decay: 1.0e-2
    scheduler_name: step
    scheduler_kwargs:
        milestones:
            - 40
        gamma: 0.1

# From ./config/default_train.yaml
# Prevents Hydra altering the working directory
hydra:
  run:
    dir: .
  output_subdir: null
  sweep:
    dir: .
    subdir: .


```

If a `TreebankDataModule` has been generated and saved already, it can be loaded in by using the override
```bash

++data.file_path=./morphological_tagging/data/corpora/{$FILE_NAME}.pickle

```

This will invalidate all other `data` keys, besides `batch_size`.

Make certain to alter the preprocessor's language as well:

```bash

++logging.logger_kwargs.job_type={$LANGUAGE} ++preprocessor.language={$LANGUAGE} ++data.language={$LANGUAGE}

```

The `trainer` key specifies keyword arguments for a PyTorch-Lightning trainer. For example, to easily specify half-precision training, simply use override

```bash
++trainer.precision=16
```
</p>
</details>
Models are save in the checkpoints directory.

### Evaluating
To evaluate a trained model on a pre-defined dataset stored in `./data/corpora`, run

```bash
python -u evaluate_tagger.py ++model_name=UDPipe2 ++dataset_name={$LANGUAGE}_{$TREEBANKNAME} evaluate_tagger.py hydra/job_logging=disabled hydra/hydra_logging=disabled
```

It will automatically search for the most recent version of UDPipe2 model available.

The eval files will be stored in `./eval`. These can be read and analyzed in the [evaluation notebook](./evaluation.ipynb)

### Pipeline Creation/Loading

# References

[^UDPipe2 CONLL]: Straka, M. (2018, October). UDPipe 2.0 prototype at CoNLL 2018 UD shared task. In Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies (pp. 197-207).

[^UDPipe2 SIGMORPHON]: Straka, M., Straková, J., & Hajič, J. (2019). UDPipe at SIGMORPHON 2019: Contextualized embeddings, regularization with morphological categories, corpora merging. arXiv preprint arXiv:1908.06931.

[^UDPipe2 EvaLatin]: Straka, M., & Straková, J. (2020). UDPipe at EvaLatin 2020: Contextualized embeddings and treebank embeddings. arXiv preprint arXiv:2006.03687.
