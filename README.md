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
    alt="This project's WandB page."
    style="float: center;"
    />
</a>
<h4 align="center">MSc AI Thesis @ University of Amsterdam</h4>
</p>

> **New**
  >1. UDIFY models added
> 2. Focus on out-of-the-box pipelines
> 3. DogTag models implemented
> 4. Script to automate tagging with CLI

> **To Do**
> 1. Add evaluation details
> 2. Add UDIFY training details
> 3. Add DogTag training details

This project has been supported by the European Union's Horizon 2020 research and innovation programme under grant agreement No 825299 ([GoURMET](https://gourmet-project.eu/)). <img src="./misc/figures/eu_flag.jpg" width="40px" style="vertical-align:middle">

This repo holds a collection of utilities and scripts for building CoNLL/UniMorph corpora, training joint morphological taggers and lemmatizers, evaluating trained models and converting models into performant pipelines.

For a downstream task, we needed SoTA morphological taggers for text in context, applicable to many languages. Unfortunately, little additional research has been performed since the [2019 SIGMORPHON/CONLL Shared Task 2](https://sigmorphon.github.io/sharedtasks/2019/task2/)<sup>[[1]](#sharedtask2019)</sup>, and existing implementations use outdated datasets or suboptimal tagging schemas. This repo instead allows for extending those implementations to the latest [UD treebanks](https://universaldependencies.org/#language-) and the the UniMorph tagging schema<sup>[[2]](#unimorphschema)</sup>, all within PyTorch and PyTorch Lightning.

<!--
### Contents

1. **Environment:**
2. **Datasets:**
3. **Models:**
4. **References:** cited papers useful for further reading
-->

# Dependencies

To install the minimal dependencies use pre-defined pipelines in Python:

```bash
pip install numpy==1.22.0
pip install pytorch==1.10.0
pip install torchtext==0.11.0
pip install transformers==4.12.5
```

To run `./tag_file.py` on untokenized text, `sacremoses` is additionally required:

```bash
pip install sacremoses==0.0.49
```

For all other functionalities, you can use the provided environment. This is quite heavy, building a full modern deep learning environment (including [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), [Weights and Biases](https://docs.wandb.ai/ref/python), [Hydra](https://hydra.cc/docs/intro/), etc.):

```bash
conda env create -f env.yaml
```

Tested on Windows 11 and Debian Linux-4.19.0.

# Basic Usage

Pipelines were designed to contain all logic necessary for tokenization, collating batches and converting output on top of pretrained PyTorch models: in short, sentences go in, lemmas and morph. tags come out. The main aim is out-of-the-box ease of use.

Currently implemented pipelines:

1. [UFAL Prague](https://github.com/ufal/udpipe/tree/udpipe-2)'s UDPipe2<sup>[[4](#udpipe2conll), [5](#UDPipe2SIGMORPHON), [6](#UDPipe2EvaLatin)]</sup>
2. Our own [CANINE](https://huggingface.co/google/canine-s)<sup>[[7](#canine)]</sup> based [DogTag](?)

<details>
<summary><b>When to use what</b></summary>
<p>
Currently, both UDPipe2 and DogTag perform roughly the same. UDPipe2 is a better morphological tagger for higher resource languages and DogTag seems to be better at both lemmatizing and tagging for low resource languages (Finnish, Turkish). Since the difference is more pronounced for the latter, and a bit of vanity, DogTag should be the default.

In case memory or speed constraints are in place:

1. **Memory**: UDPipe requires loading in both word and contextual (i.e. a BERT variant) embeddings. These dominate memory used. DogTag requires only loading in a smaller transformer, CANINE. For both file and RAM usage, DogTag is significantly slimmer (~1.5 GB).
2. **Inference Speed**: CANINE operates at the character level, resulting in far larger input strings. As such, it is quite a bit faster at equal batch sizes than DogTag.
3. **Training Speed**: UDPipe requires finetuning a relatively small number of parameters on top of a lot of pre-trained modules. Training is *much* faster than other implemented models.

</p>
</details>

## Command-line Interface

The `tag_file.py` script allows you to quickly tag a text file of sentences into some other format containing lemmas and morphological features. To tag a file, the path and language (entered as natural text) must be provided, e.g. on CPU:

```bash
python tag_file.py --file_path {$FILE_TO_TAG} --language {$LANGUAGE} --gpu 0
```

Additional command line options include:
```txt
  --file_path FILE_PATH   the location of the text file
  --language LANGUAGE     the language of the text
  --pipeline PIPELINE     pipeline checkpoint name in './pipelines', must contain architecture
  --gpu GPU               whether to annotate on GPU, if available
  --batch_size BATCH_SIZE number of lines being fed into the pipeline
  --encoding ENCODING     encoding of text file
  --output_format {single_pickle_file,separate_text_files,single_jsonlines_file} [{single_pickle_file,separate_text_files,single_jsonlines_file} ...] output format
```

## Python API

Once created, the pipelines can be saved and loaded without needing to point to a dataset class or a model checkpoint. The checkpoint files contain only a minimal subset of the parameters needed for the pipeline, and are thus smaller than the full model at run time.

```python
from morphological_tagging.pipelines import UDPipe2Pipeline, DogTagPipeline

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

# Training & Further Development
## Datasets

The files used for the SIGMORPHON/CONLL Shared Task 2019 are in CONLL-U format, except with the `features` column automatically converted<sup>[3](#udconversion)</sup> to UniMorph tagsets<sup>[2](#unimorphschema)</sup>. Data files can be found [here](https://github.com/sigmorphon/2019).

Since the competition, the UD treebanks have seen 6 new releases, with improvements and extensions made to many of the used corpora. To use these datasets instead, we have similary converted UD2.9 to carry UniMorph tags using the [ud-compatibility](https://github.com/unimorph/ud-compatibility) repo. Datasets with existing train/valid/test splits can be found in the [Google Drive](https://drive.google.com/file/d/1lSYGYB-4b5dztlg1iilccctI1KAxVV_e/view?usp=sharing).

For more details regarding dataset creation and usage, see [here](./morphological_tagging/README.md).

## Models & Training

All pre-trained models in pipeline format (including dictionaries to allow mapping from raw text to input and output to processed text) can be found in the Google Drive folder [here](https://drive.google.com/drive/u/0/folders/1O0NZgyjkiuWQ9FuqZpsgFII2j8487Mct).

An early design choice was to opt for seq-first batching for UDPipe2, but batch-first for others. This make dataset files incompatible between models, unfortunately.

For details, see [here](./morphological_tagging/README.md). Generally, for deep learning experiments no reproduction is exact, and this project is no exception. Differences are detailed for each model in their respective section, ordered by expected impact (largest to smallest). Furthermore, test set performance is reported.

## Evaluating

**TODO**: detailed evaluation of models (also extended beyond CoNLL/SIGMORPHON 2019).
<!--
To evaluate a trained model on a pre-defined dataset stored in `./data/corpora`, run

```bash
python -u evaluate_tagger.py ++model_name=UDPipe2 ++dataset_name={$LANGUAGE}_{$TREEBANKNAME} hydra/job_logging=disabled hydra/hydra_logging=disabled
```

It will automatically search for the most recent version of UDPipe2 model available.

The eval files will be stored in `./eval`. These can be read and analyzed in the [evaluation notebook](./evaluation.ipynb)
-->
# References

<a name="sharedtask2019">1</a>: McCarthy, A. D., Vylomova, E., Wu, S., Malaviya, C., Wolf-Sonkin, L., Nicolai, G., Kirov, C., Silfverberg, M., Mielke, S. J., Heinz, J., Cotterell, R. & Hulden, M. (2019). The SIGMORPHON 2019 shared task: Morphological analysis in context and cross-lingual transfer for inflection. arXiv preprint arXiv:1910.11493.

<a name="unimorphschema">2</a>: Sylak-Glassman, J. (2016). The composition and use of the universal morphological feature schema (unimorph schema). Johns Hopkins University.

<a name="udconversion">3</a>: McCarthy, A. D., Silfverberg, M., Cotterell, R., Hulden, M., & Yarowsky, D. (2018). Marrying universal dependencies and universal morphology. arXiv preprint arXiv:1810.06743.

<a name="udpipe2conll">4</a>: Straka, M. (2018, October). UDPipe 2.0 prototype at CoNLL 2018 UD shared task. In Proceedings of the CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies (pp. 197-207).

<a name="UDPipe2SIGMORPHON">5</a>: Straka, M., Straková, J., & Hajič, J. (2019). UDPipe at SIGMORPHON 2019: Contextualized embeddings, regularization with morphological categories, corpora merging. arXiv preprint arXiv:1908.06931.

<a name="UDPipe2EvaLatin">6</a>: Straka, M., & Straková, J. (2020). UDPipe at EvaLatin 2020: Contextualized embeddings and treebank embeddings. arXiv preprint arXiv:2006.03687.

<a name="canine">7</a>: Clark, J. H., Garrette, D., Turc, I., & Wieting, J. (2022). Canine: Pre-training an efficient tokenization-free encoder for language representation. Transactions of the Association for Computational Linguistics, 10, 73-91.
