<h1 align="center">
  Morphological Tagging and Lemmatization in Context
</h1>

<p align="center">
<!--
<a href="https://www.notion.so/MSc-AI-Thesis-9c3ba8027f6b4e3a82f0e391a6db76a9">
    <img
    src="https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white"
    alt="This project's Notion page."
    style="float: center;"
    />
</a>
-->
<a href="https://wandb.ai/verhivo/morph_tag_lemmatize?workspace=user-verhivo">
    <img src="https://img.shields.io/badge/WandB-%23000000.svg?&style=for-the-badge&logo=weightsandbiases&logoColor=#FFBE00"
    alt="This project's WandB page."
    style="float: center;"
    />
</a>
<a href="https://drive.google.com/drive/folders/1O0NZgyjkiuWQ9FuqZpsgFII2j8487Mct?usp=sharing">
    <img src="https://img.shields.io/badge/Drive-%23000000.svg?&style=for-the-badge&logo=googledrive&logoColor=#FFBE00"
    alt="Model checkpoints and datasets"
    style="float: center;"
    />
</a>
<a href="https://github.com/ioverho/morph_tag_lemmatize/blob/main/misc/msc_thesis_chp_1.pdf">
    <img src="https://img.shields.io/badge/thesis-%23000000.svg?&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KCjxzdmcKICAgZmlsbD0iIzAwMDAwMCIKICAgdmVyc2lvbj0iMS4xIgogICBpZD0iWE1MSURfMzhfIgogICB2aWV3Qm94PSIwIDAgMjQgMjQiCiAgIHhtbDpzcGFjZT0icHJlc2VydmUiCiAgIHNvZGlwb2RpOmRvY25hbWU9ImRvY3VtZW50LXBkZi1zdmdyZXBvLWNvbS5zdmciCiAgIGlua3NjYXBlOnZlcnNpb249IjEuMi4xICg5YzZkNDFlNDEwLCAyMDIyLTA3LTE0KSIKICAgeG1sbnM6aW5rc2NhcGU9Imh0dHA6Ly93d3cuaW5rc2NhcGUub3JnL25hbWVzcGFjZXMvaW5rc2NhcGUiCiAgIHhtbG5zOnNvZGlwb2RpPSJodHRwOi8vc29kaXBvZGkuc291cmNlZm9yZ2UubmV0L0RURC9zb2RpcG9kaS0wLmR0ZCIKICAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICB4bWxuczpzdmc9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcwogICBpZD0iZGVmczMwIiAvPjxzb2RpcG9kaTpuYW1lZHZpZXcKICAgaWQ9Im5hbWVkdmlldzI4IgogICBwYWdlY29sb3I9IiNmZmZmZmYiCiAgIGJvcmRlcmNvbG9yPSIjMTExMTExIgogICBib3JkZXJvcGFjaXR5PSIxIgogICBpbmtzY2FwZTpzaG93cGFnZXNoYWRvdz0iMCIKICAgaW5rc2NhcGU6cGFnZW9wYWNpdHk9IjAiCiAgIGlua3NjYXBlOnBhZ2VjaGVja2VyYm9hcmQ9IjEiCiAgIGlua3NjYXBlOmRlc2tjb2xvcj0iI2QxZDFkMSIKICAgc2hvd2dyaWQ9ImZhbHNlIgogICBpbmtzY2FwZTp6b29tPSIzNC40MTY2NjciCiAgIGlua3NjYXBlOmN4PSIxMi4wMTQ1MjgiCiAgIGlua3NjYXBlOmN5PSIxMi4wMTQ1MjgiCiAgIGlua3NjYXBlOndpbmRvdy13aWR0aD0iMTkyMCIKICAgaW5rc2NhcGU6d2luZG93LWhlaWdodD0iMTAxMyIKICAgaW5rc2NhcGU6d2luZG93LXg9Ii05IgogICBpbmtzY2FwZTp3aW5kb3cteT0iLTkiCiAgIGlua3NjYXBlOndpbmRvdy1tYXhpbWl6ZWQ9IjEiCiAgIGlua3NjYXBlOmN1cnJlbnQtbGF5ZXI9IlhNTElEXzM4XyIgLz4KPGcKICAgaWQ9ImRvY3VtZW50LXBkZiIKICAgc3R5bGU9ImZpbGw6I2ZmZmZmZiI+Cgk8ZwogICBpZD0iZzQiCiAgIHN0eWxlPSJmaWxsOiNmZmZmZmYiPgoJCTxwYXRoCiAgIGQ9Ik0xMSwyMEg3di04aDRjMS42LDAsMywxLjUsMywzLjJ2MS42QzE0LDE4LjUsMTIuNiwyMCwxMSwyMHogTTksMThoMmMwLjUsMCwxLTAuNiwxLTEuMnYtMS42YzAtMC42LTAuNS0xLjItMS0xLjJIOVYxOHogICAgIE0yLDIwSDB2LThoM2MxLjcsMCwzLDEuMywzLDNzLTEuMywzLTMsM0gyVjIweiBNMiwxNmgxYzAuNiwwLDEtMC40LDEtMXMtMC40LTEtMS0xSDJWMTZ6IgogICBpZD0icGF0aDIiCiAgIHN0eWxlPSJmaWxsOiNmZmZmZmYiIC8+Cgk8L2c+Cgk8ZwogICBpZD0iZzgiCiAgIHN0eWxlPSJmaWxsOiNmZmZmZmYiPgoJCTxyZWN0CiAgIHg9IjE1IgogICB5PSIxMiIKICAgd2lkdGg9IjYiCiAgIGhlaWdodD0iMiIKICAgaWQ9InJlY3Q2IgogICBzdHlsZT0iZmlsbDojZmZmZmZmIiAvPgoJPC9nPgoJPGcKICAgaWQ9ImcxMiIKICAgc3R5bGU9ImZpbGw6I2ZmZmZmZiI+CgkJPHJlY3QKICAgeD0iMTUiCiAgIHk9IjEyIgogICB3aWR0aD0iMiIKICAgaGVpZ2h0PSI4IgogICBpZD0icmVjdDEwIgogICBzdHlsZT0iZmlsbDojZmZmZmZmIiAvPgoJPC9nPgoJPGcKICAgaWQ9ImcxNiIKICAgc3R5bGU9ImZpbGw6I2ZmZmZmZiI+CgkJPHJlY3QKICAgeD0iMTUiCiAgIHk9IjE2IgogICB3aWR0aD0iNSIKICAgaGVpZ2h0PSIyIgogICBpZD0icmVjdDE0IgogICBzdHlsZT0iZmlsbDojZmZmZmZmIiAvPgoJPC9nPgoJPGcKICAgaWQ9ImcyMCIKICAgc3R5bGU9ImZpbGw6I2ZmZmZmZiI+CgkJPHBvbHlnb24KICAgcG9pbnRzPSIyNCwyNCA0LDI0IDQsMjIgMjIsMjIgMjIsNi40IDE3LjYsMiA2LDIgNiw5IDQsOSA0LDAgMTguNCwwIDI0LDUuNiAgICIKICAgaWQ9InBvbHlnb24xOCIKICAgc3R5bGU9ImZpbGw6I2ZmZmZmZiIgLz4KCTwvZz4KCTxnCiAgIGlkPSJnMjQiCiAgIHN0eWxlPSJmaWxsOiNmZmZmZmYiPgoJCTxwb2x5Z29uCiAgIHBvaW50cz0iMjMsOCAxNiw4IDE2LDIgMTgsMiAxOCw2IDIzLDYgICAiCiAgIGlkPSJwb2x5Z29uMjIiCiAgIHN0eWxlPSJmaWxsOiNmZmZmZmYiIC8+Cgk8L2c+CjwvZz4KPC9zdmc+Cg==
&style=for-the-badge&logoColor=#FFFFFF"
    alt="Relevant thesis chapter."
    style="float: center;"
    />
</a>
<h4 align="center">MSc AI Thesis @ University of Amsterdam</h4>
</p>

> **New**
> 1. UDIFY models added
> 2. Evaluation procedure
> 3. Added documentation & thesis chapter

> **To Do**
> 1. Dump additional information in `./morphological_tagging/README.md`

This project has been supported by the European Union's Horizon 2020 research and innovation programme under grant agreement No 825299 ([GoURMET](https://gourmet-project.eu/)). <img src="./misc/figures/eu_flag.jpg" width="40px" style="vertical-align:middle">

This repo holds a collection of utilities and scripts for building CoNLL/UniMorph corpora, training joint morphological taggers and lemmatizers, evaluating trained models and converting models into performant pipelines.

We replicate the winning systems from the [2019 SIGMORPHON/CONLL Shared Task 2](https://sigmorphon.github.io/sharedtasks/2019/task2/)<sup>[[1]](#sharedtask2019)</sup>, extended to the latest [UD treebanks](https://universaldependencies.org/#language-) and the the UniMorph tagging schema<sup>[[2]](#unimorphschema)</sup>. We further contribute a competitive architecture of our own. We further extend the evaluation method, and show robust performance on many language types for both known and unknown word-forms/lemmas.

Uses a single, consistent PyTorch framework. All models are easy to train and test, and once trained, to use.

<!--
### Contents

1. **Environment:**
2. **Datasets:**
3. **Models:**
4. **References:** cited papers useful for further reading
-->

# Installation

This repo contains code for both development and inference. The latter requires far fewer external dependencies than the former.

1. Clone this repo, e.g.

```bash
git clone https://github.com/ioverho/morph_tag_lemmatize.git
cd morph_tag_lemmatize
```

2. Build the Anaconda environment

```bash
conda env create -f env.yaml # For inference only
conda env create -f env_development.yaml # For inference & dev
conda activate morph_tag_lemmatize
```

3. Download needed pipeline checkpoints or datasets

Tested on Windows 11, Debian Linux 4.19.0 and Ubuntu 20.04.5 (through WSL2).

# Inference

All pre-trained models (including dictionaries to allow mapping from raw text to input and output to processed text) can be found in the Google Drive folder [here](https://drive.google.com/drive/u/0/folders/1O0NZgyjkiuWQ9FuqZpsgFII2j8487Mct). The languages chosen are meant to represent a broad set of morphologically interesting European languages, and is by no means complete. Detailed below is how to train new models on other languages.

## Basic Usage

Pipelines were designed to contain all logic necessary for tokenization, collating batches and converting output on top of pretrained PyTorch models: in short, sentences go in, lemmas and morph. tags come out. The main aim is out-of-the-box ease of use.

Currently implemented pipelines:

1. [UFAL Prague](https://github.com/ufal/udpipe/tree/udpipe-2)'s UDPipe2<sup>[[4](#udpipe2conll), [5](#UDPipe2SIGMORPHON), [6](#UDPipe2EvaLatin)]</sup>
2. [Dan Kondratyuk](https://github.com/Hyperparticle/udify)'s UDIFY<sup>[[8](#udify), [9](#udifysigmorphon)]</sup>
3. Our own [CANINE](https://huggingface.co/google/canine-s)<sup>[[7](#canine)]</sup> based [DogTag](?)

Both UDIFY and DogTag support multilingual pre-training. This can provide a modest performance boost, at the cost of significantly more expensive training. In the saved model checkpoints, `mono` indicates training only on the target language (i.e. no pre-training), and `multi` indicates first training on all languages from the same typological family before fine-tuning on the target language.

<details>
<summary><b>When to use what</b></summary>
<p>
All models perform roughly the same, overall. UDIFY works best, especially with multilingual pre-training, even on lower resource languages. DogTag is a very strong lemmatizer, but lags somewhat on morphological tagging. It also does not benefit from multilingual pre-training. UDPipe tends to perform worse.

In case memory or speed constraints are in place:

1. **Memory**: UDPipe requires loading in both word and contextual (i.e. a BERT variant) embeddings. These dominate memory used. DogTag requires only loading in a smaller transformer, CANINE. For both file and RAM usage, DogTag is significantly slimmer (~1.5 GB).
2. **Inference Speed**: CANINE operates at the character level, resulting in far larger input strings. UDIFY operates at the BPE level, but uses two, separate LSTMs for each task. As such, UDPipe has higher troughput. However, all models are reasonably fast on both CPU and GPU.
3. **Training Speed**: UDPipe requires finetuning a relatively small number of parameters on top of a lot of pre-trained modules. Training is *much* faster than other implemented models.

</p>
</details>

## Command-line Interface

The `tag_file.py` script allows you to quickly tag a text file of sentences into some other format containing lemmas and morphological features. To tag a file, the path and language (entered as natural text) must be provided, e.g. on CPU:

```bash
python tag_file.py \
  --file_path {$FILE_TO_TAG} \
  --language {$LANGUAGE} \
  --gpu 0

```

Additional command line options include:
```txt
  --file_path     the location of the text file
  --language      the language of the text
  --pipeline_dir  location of the pretrained pipelines. Defaults to './pipelines'
  --pipeline      pipeline checkpoint name in `pipeline_dir`, must contain architecture
  --gpu           whether to annotate on GPU, if available
  --batch_size    number of lines being fed into the pipeline
  --encoding      encoding of text file
  --output_format {single_pickle_file,separate_text_files,single_jsonlines_file} output format
```

## Python API

Once created, the pipelines can be saved and loaded without needing to point to a dataset class or a model checkpoint. The checkpoint files contain only a minimal subset of the parameters needed for the pipeline, and are thus smaller than the full model at run time.

```python
from morphological_tagging.pipelines import UDPipe2Pipeline, UDIFYPipeline, DogTagPipeline

pipeline = PipelineClass.load(save_loc, map_location=device)
```

To use, simply feed in a list of strings, or a list of lists if tokenization occurs outside of tagger.

```python
# If tokenizer is provided
pipeline(List[str])

# If tokenization is performed already
pipeline(List[List[str]], is_pre_tokenized=True)

# If sampling from TreebankDataModule
pipeline(Tuple[Union[List, torch.Tensor]], is_batch_input=True)
```

Default output is a tuple of lists containing, in order, the predicted lemmas, lemma scripts, morphological tags and categories. To change to a per token collection, use the transpose argument in the forward call:

```python
pipeline(List[str], transpose=True) -> List[Tuple[lemma, lemma_script, morph_tags, morph_cats], ...]
```

For more details regarding pipeline creation, see [here](./morphological_tagging/README.md).

# Training & Evaluation
## Datasets

The files used for the SIGMORPHON/CONLL Shared Task 2019 are in CONLL-U format, except with the `features` column automatically converted<sup>[[3]](#udconversion)</sup> to UniMorph tagsets<sup>[[2]](#unimorphschema)</sup>. The original data files can be found [here](https://github.com/sigmorphon/2019).

Since the competition, the UD treebanks have seen 6 new releases, with improvements and extensions made to many corpora. We have similary converted UD2.9 to carry UniMorph tags using the [ud-compatibility](https://github.com/unimorph/ud-compatibility) repo. Datasets with existing train/valid/test splits can be found in the [Google Drive](https://drive.google.com/file/d/1lSYGYB-4b5dztlg1iilccctI1KAxVV_e/view?usp=sharing).

While many more languages are available, we identified 38 which contain high quality annotations and enough samples for succesful training. Dataset size and diversity still varies considerably. For more details regarding dataset creation and usage, see [here](./morphological_tagging/README.md).

Treebanks can be created via the ``build_treebank_corpus.py`` file. It looks for all datasets from a language, and merges those that meet certain criteria. Finally, batches are created for fast training. An annotated config file can be found in [`./morphological_tagging/config/treebank_corpus.yaml`](./morphological_tagging/config/treebank_corpus.yaml).

## Models & Training

An early design choice was to opt for seq-first batching for UDPipe2, but batch-first for others. This make dataset files incompatible between models, unfortunately.

For details, see [here](./morphological_tagging/README.md). Generally, for deep learning experiments no reproduction is exact, and this project is no exception. Differences are detailed for each model in their respective section, ordered by expected impact (largest to smallest). Furthermore, test set performance is reported.

All training is conducted through the ``train_tagger.py`` script. A config file needs to be supplied to [Hydra](https://hydra.cc/) when calling through CLI. For example,

```bash
python train_tagger.py \
  --config-name udify_experiment \
  ++trainer precision=16 \
  gpu=1 \
  hydra/job_logging=disabled \
  hydra/hydra_logging=disabled

```

trains a UDIFY model on the default dataset using half-precision on GPU. The last two lines disable hydra job-logging, which is strongly recommended is using a third-party logger like wandb or tensorboard. Additional configuration options can be found in the respective `/config/model` and `/config/data` directories, with some default values taken from `/config/default_train.yaml`.

## Evaluating

To evaluate a trained model (checkpoint in `./morphological_tagging/checkpoints/` on a pre-defined dataset stored in `./morphological_tagging/data/corpora`, run the `evaluate_tagger.py` script. The corresponding configuration file can be found under `./morphological_tagging/config/eval.yaml`.

For example, to evaluate a pretrained model `MODEL` on a dataset of `LANGUAGE`/`TREEBANKNAME` combination on GPU:

```bash

python evaluate_tagger.py \
  ++model_name={$MODEL} \
  ++dataset_name={$LANGUAGE}_{$TREEBANKNAME} \
  gpu=1 \
  hydra/job_logging=disabled hydra/hydra_logging=disabled

```

It will automatically search for the most recent version model available.

The script outputs an pickle file containing tuples of:

```txt
(token, lemma, predicted lemma, lemma script, predicted lemma script, morphological tags, predicted morphological tags, whether token is present in vocab, whether lemma is present in vocab)
```

which can be used for analysis of model behaviour. For an example, see [`evaluation.ipynb`](./evaluation.ipynb). Results from this notebook are also used in the corresponding thesis chapter. This repo comes with many eval files already produced, see [`./eval/`](./eval/)

The script also conveniently builds a pipeline object from the checkpoint (saved in `./pipelines`). The pipeline contains the aggregated performance stats printed at the end of this script.

# References

<a name="sharedtask2019">1</a>: McCarthy, A. D., Vylomova, E., Wu, S., Malaviya, C., Wolf-Sonkin, L., Nicolai, G., Kirov, C., Silfverberg, M., Mielke, S. J., Heinz, J., Cotterell, R. & Hulden, M. (2019). The SIGMORPHON 2019 shared task: Morphological analysis in context and cross-lingual transfer for inflection. arXiv preprint arXiv:1910.11493.

<a name="unimorphschema">2</a>: Sylak-Glassman, J. (2016). The composition and use of the universal morphological feature schema (unimorph schema). Johns Hopkins University.

<a name="udconversion">3</a>: McCarthy, A. D., Silfverberg, M., Cotterell, R., Hulden, M., & Yarowsky, D. (2018). Marrying universal dependencies and universal morphology. arXiv preprint arXiv:1810.06743.

<a name="udpipe2conll">4</a>: Straka, M. (2018, October). UDPipe 2.0 prototype at CoNLL 2018 UD shared task. In Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies (pp. 197-207).

<a name="UDPipe2SIGMORPHON">5</a>: Straka, M., Straková, J., & Hajič, J. (2019). UDPipe at SIGMORPHON 2019: Contextualized embeddings, regularization with morphological categories, corpora merging. arXiv preprint arXiv:1908.06931.

<a name="UDPipe2EvaLatin">6</a>: Straka, M., & Straková, J. (2020). UDPipe at EvaLatin 2020: Contextualized embeddings and treebank embeddings. arXiv preprint arXiv:2006.03687.

<a name="canine">7</a>: Clark, J. H., Garrette, D., Turc, I., & Wieting, J. (2022). Canine: Pre-training an efficient tokenization-free encoder for language representation. Transactions of the Association for Computational Linguistics, 10, 73-91.

<a name="udify">8</a>: Kondratyuk, D., & Straka, M. (2019). 75 languages, 1 model: Parsing universal dependencies universally. arXiv preprint arXiv:1904.02099.

<a name="udifysigmorphon">9</a>: Kondratyuk, D. (2019, August). Cross-lingual lemmatization and morphology tagging with two-stage multilingual BERT fine-tuning. In Proceedings of the 16th Workshop on Computational Research in Phonetics, Phonology, and Morphology (pp. 12-18).
