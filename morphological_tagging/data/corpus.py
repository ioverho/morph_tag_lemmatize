import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import product
from typing import Optional, List, Tuple, Dict, Union
import pickle

import omegaconf
import conllu
from conllu import parse_incr
import pytorch_lightning as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset
import torchtext
from torchtext.vocab import build_vocab_from_iterator, Vectors
from morphological_tagging.data.dataloaders import TokenDataloader

from morphological_tagging.data.lemma_script import LemmaScriptGenerator


BASEPATH = "./morphological_tagging/data/sigmorphon_2019/task2"

ALL_TAGS_FP = "./morphological_tagging/data/uni_morph_tags.json"

with open("./morphological_tagging/data/treebank_metadata.json") as f:
    TREEBANK_METADATA = json.load(f)

with open("./morphological_tagging/data/family_to_language.json") as f:
    FAMILY_TO_LANGUAGE = json.load(f)

with open("./morphological_tagging/data/language_to_family.json") as f:
    LANGUAGE_TO_FAMILY = json.load(f)


def get_conllu_files(
    language: str, name: str = "merge", source: str = None
) -> Union[dict[list], list]:
    """Get the UD/CONLLU/SIGMORPHON treebanks from the source dir.
    Only gets the `.conllu files` and returns absolute paths.

    Args:
        language (str): the desired language, or all for all languages in the directory
        name (str): the desired treebank name, or merge for all. Default to "merge"

    Returns:
        List[Tuple[str, str, str, str]]: a list of tuples containing, in order, the absolute file path,
        the treebank's name, the split (train/dev/test) and the language
    """

    if source is None:
        source = BASEPATH

    files = []
    for f in Path(source).glob("**/*.conllu"):
        t_lang, t_name = f.parent.stem[3:].split("-", maxsplit=1)

        if (
            # check for language match
            (t_lang.lower() == language.lower() or language.lower() == "all")
            and
            # check for treebank name match
            (name.lower() == t_name.lower() or name.lower() == "merge")
            and
            # check if file is not "covered" (features removed for test set)
            ("covered" not in f.parts[-1])
        ):
            split = f.stem.split("-")[-1]

            files.append((str(f.absolute()), t_name, split, t_lang))

    if len(files) == 0:
        print(f"No files found for {language}_{name} in {source}.")

    return files


@dataclass
class Tree:
    """A class for holding a single tree.
    """

    raw: List = field(default_factory=lambda: [])
    tokens: List = field(default_factory=lambda: [])
    lemmas: List = field(default_factory=lambda: [])
    morph_tags: List = field(default_factory=lambda: [])

    def add_parsed(self, word_form, lemma, morph_tags):

        self.raw.append((word_form, lemma, morph_tags))
        self.tokens.append(word_form)
        self.lemmas.append(lemma)
        self.morph_tags.append(morph_tags)

    def add(self, branch: List):
        _, word_form, lemma, _, _, morph_tags, _, _, _, _ = branch

        self.raw.append((word_form, lemma, morph_tags))
        self.tokens.append(word_form)
        self.lemmas.append(lemma)
        self.morph_tags.append(morph_tags.rsplit(";"))

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i: int):
        return self.raw[i]

    def __str__(self):
        return f"Tree({self.raw})"

    def __repr__(self):
        return self.__str__()


@dataclass
class Document:
    """A class for holding a single text document.
    """

    sent_id: Optional[str] = None
    split: Optional[str] = None
    treebank: Optional[str] = None
    language: Optional[str] = None
    text: Optional[str] = None
    tree: List = field(default_factory=lambda: Tree())

    def __str__(self):
        return f"Doc(sent_id={self.sent_id})"

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.tree.tokens)

    @property
    def tokens(self):
        return self.tree.tokens

    @property
    def lemmas(self):
        return self.tree.lemmas

    @property
    def morph_tags(self):
        return self.tree.morph_tags

    def set_tensors(
        self, chars_tensor, tokens_tensor, morph_tags_tensor, morph_cats_tensor
    ):

        self.chars_tensor = chars_tensor

        self.tokens_tensor = tokens_tensor

        self.morph_tags_tensor = morph_tags_tensor

        self.morph_cats_tensor = morph_cats_tensor

    def set_morph_cats(self, cats_list: List):

        self.morph_cats = cats_list

    def set_lemma_tags(self, tags_tensor: torch.LongTensor):

        self.lemma_tags_tensor = tags_tensor

    # ! Deprecated method
    # TODO (ivo): remove
    def set_word_embeddings(
        self, word_emb: torch.Tensor, word_vec_type: str = "Undefined"
    ):

        self.word_emb_type = word_vec_type
        self.word_emb = word_emb

    # ! Deprecated method
    # TODO (ivo): remove
    def set_context_embeddings(
        self, context_emb: torch.Tensor, model_name: str = "Undefined"
    ):

        self.model_name = model_name
        self.context_emb = context_emb

    def set_pretrained_embeddings(self, x: torch.Tensor) -> None:

        self._pretrained_embeddings = x

    @property
    def pretrained_embeddings(self):

        if hasattr(self, "_pretrained_embeddings"):
            return self._pretrained_embeddings
        else:
            return 0


@dataclass
class DocumentCorpus(Dataset):
    """A class for reading, holding and processing many documents.
    """

    docs: List = field(default_factory=lambda: [])
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"
    treebanks: Dict = field(default_factory=lambda: defaultdict(int))
    splits: Dict = field(default_factory=lambda: defaultdict(list))
    batch_first: bool = False
    sorted: bool = True
    return_tokens_raw: bool = False
    max_tokens: int = 256
    max_chars: int = 2048
    remove_unique_lemma_scripts: bool = False

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i: int):
        return self.docs[i]

    def __str__(self):
        return f"DocumentCorpus(\n\tlen={len(self.docs)},\n\ttreebanks={self.treebanks.keys()},\n\tsplits={self.splits.keys()},\n\tbatch_first={self.batch_first},\n\tsorted={self.sorted}\n)"

    def __repr__(self):
        return self.__str__()

    def clear_docs(self):

        self.docs = []

    def parse_connlu_file(
        self, fp, split: str, name: str, language: str, remove_duplicates: bool = True
    ):
        """[summary]

        Args:
            fp ([type]): [description]
            split (str): [description]
            name (str): [description]
            language (str): [description]
            remove_duplicates (bool, optional): [description]. Defaults to True.
        """

        treebank_docs = []

        with open(fp, "r", encoding="utf-8") as f:
            for _, tokenlist in enumerate(
                parse_incr(
                    f,
                    fields=[
                        "id",
                        "form",
                        "lemma",
                        "upos",
                        "xpos",
                        "feats",
                        "head",
                        "deprel",
                        "deps",
                        "misc",
                    ],
                    field_parsers={"feats": lambda line, i: line[i].split(";")},
                )
            ):

                tree = Tree()

                for token in tokenlist:

                    if not isinstance(token["id"], int):
                        # Throw away all empty nodes and multi-token words
                        continue

                    else:
                        # Add to the tree the parsed attributes
                        tree.add_parsed(token["form"], token["lemma"], token["feats"])

                if (
                    len(tokenlist) > self.max_tokens
                    or len(tokenlist.metadata["text"]) > self.max_chars
                ):
                    continue
                else:
                    doc = Document(
                        sent_id=tokenlist.metadata["sent_id"],
                        text=tokenlist.metadata["text"],
                        split=split,
                        language=language,
                        treebank=name,
                        tree=tree,
                    )

                    treebank_docs.append(doc)

            if remove_duplicates:
                treebank_docs = list({d.text: d for d in treebank_docs}.values())

        self.docs.extend(treebank_docs)

    def _set_lemma_tags(self):

        self.script_counter, self.script_examples = Counter(), defaultdict(set)
        docs_scripts, docs_scripts_by_treebank = [], defaultdict(list)
        for i, d in enumerate(self.docs):

            doc_scripts = []
            for wf, lm in zip(d.tokens, d.lemmas):

                lemma_script = LemmaScriptGenerator(wf, lm).get_lemma_script()
                self.script_counter[lemma_script] += 1

                doc_scripts.append(lemma_script)

                if len(self.script_examples[lemma_script]) < 3:
                    self.script_examples[lemma_script].add(f"{wf}\u2192{lm}")

            docs_scripts.append(doc_scripts)
            docs_scripts_by_treebank[d.treebank].extend(doc_scripts)

        docs_scripts_by_treebank = dict(docs_scripts_by_treebank)

        docs_scripts_by_treebank = {
            k: set(v) for k, v in docs_scripts_by_treebank.items()
        }

        # Remove scripts of irregulars that occur only in one treebank
        if self.remove_unique_lemma_scripts:
            self.scripts_disallowed = {
                r
                for r, c in Counter(
                    [
                        v
                        for sublist in docs_scripts_by_treebank.values()
                        for v in sublist
                    ]
                ).items()
                if ("ign" in r and c <= 1)
            }
        else:
            self.scripts_disallowed = {}

        # Generate script to class conversion
        self.script_to_id = {
            k: i
            for i, (k, _) in enumerate(
                sorted(self.script_counter.items(), key=lambda x: x[1], reverse=True)
            )
            if k not in self.scripts_disallowed
        }

        self.id_to_script = list(self.script_to_id.keys())

        self.script_counter = Counter(
            {k: self.script_counter[k] for k in self.script_to_id.keys()}
        )
        self.script_examples = {
            k: self.script_examples[k] for k in self.script_to_id.keys()
        }

        ## Add the scripts as classes to the individual documents
        for i, doc_scripts in enumerate(docs_scripts):

            self.docs[i].set_lemma_tags(
                torch.tensor(
                    [
                        self.script_to_id[script]
                        if script not in self.scripts_disallowed
                        else -1
                        for script in doc_scripts
                    ],
                    dtype=torch.long,
                )
            )

    def _get_vocabs(self):

        # Input vocab is defined based on the train set
        # Reduces vocab sizes and allows for testing behaviour of
        # <UNK> input at validation/test/prediction time
        train_docs = [self.docs[i] for i in self.splits["train"]]

        self.token_vocab = build_vocab_from_iterator(
            [[t for t in d.tokens] for d in train_docs],
            specials=[self.unk_token, self.pad_token],
            special_first=True,
        )
        self.token_vocab.set_default_index(self.token_vocab[self.unk_token])

        self.char_vocab = build_vocab_from_iterator(
            [[c for c in t] for d in train_docs for t in d.tokens],
            specials=[self.unk_token, self.pad_token],
            special_first=True,
        )
        self.char_vocab.set_default_index(self.token_vocab[self.unk_token])

        # Class vocabs are defined based on the all available data
        # Need the validation and test set to have sensible classes that the model
        # Could theoretically predict into
        self.morph_tag_vocab = {
            k: v
            for v, k in enumerate(
                sorted(
                    {
                        tag
                        for d in self.docs
                        for tagset in d.morph_tags
                        for tag in tagset
                    }
                )
            )
        }

        # ======================================================================
        # Morphological feature categories (for regularization)
        # ======================================================================

        all_present_feats = {
            tags for d in self.docs for seq_tags in d.morph_tags for tags in seq_tags
        }

        with open(ALL_TAGS_FP, "rb") as f:
            uni_morph_tags = json.load(f)

        # Build initial morph tag to morph category mapping from file
        self.morph_tag_cat_vocab = {
            feat: cat for feat, [_, cat] in uni_morph_tags.items()
        }

        permissible_feats = set(uni_morph_tags.keys())

        # Get set of features without direct category mapping
        unmatched_feats = set(
            f.lower() for f in all_present_feats if f.lower() not in permissible_feats
        )

        unmatched_feats_lcs = dict()
        for feat_um, feat_p in product(unmatched_feats, permissible_feats):
            # For all combinations of unmatched features and permissible features
            # get the longest common substring match between the feature and any permissible feature
            # Collect the set of morph. categories belonging to the LCS matches
            # If the set is of length 1 (i.e. all possible matches have the same category) set this as
            # the tag of the unmatched feature

            feat_p_cat = uni_morph_tags[feat_p][1]

            match = SequenceMatcher(None, feat_um, feat_p).find_longest_match(
                0, len(feat_um), 0, len(feat_p)
            )

            cur_len, cur_tag, cur_cat = unmatched_feats_lcs.get(feat_um, (0, [], set()))

            if match.size > cur_len:
                unmatched_feats_lcs[feat_um] = (match.size, [feat_p], {feat_p_cat})
            elif match.size == cur_len:
                unmatched_feats_lcs[feat_um] = (
                    cur_len,
                    cur_tag + [feat_p],
                    cur_cat.union({feat_p_cat}),
                )
            else:
                pass

        # Construct temporary mapping from unmatched features to potential category
        un_matched_feats_to_tag = {
            feat_um: (list(possible_cat)[0] if len(possible_cat) == 1 else "_")
            for feat_um, (_, _, possible_cat) in unmatched_feats_lcs.items()
        }

        self.morph_tag_cat_vocab.update(un_matched_feats_to_tag)

        # Add exception for multi features (e.g. {1/2/3}, {act/caus})
        # Get the set of categories for each feature
        # Again, if the length of the set is 1, use the category for the multi-feature
        # If not, match to "_" category
        multi_feats = {
            feat: re.split(r"[{}/]", feat[1:-1])
            for feat in unmatched_feats
            if ("{" in feat and "}" in feat)
        }
        multi_feats_cat_dict = {
            m_feat: {self.morph_tag_cat_vocab[feat] for feat in multi_feats[m_feat]}
            for m_feat in multi_feats.keys()
        }
        multi_feats_cat_dict = {
            feat: (list(cats)[0] if len(cats) == 1 else "_")
            for feat, cats in multi_feats_cat_dict.items()
        }

        self.morph_tag_cat_vocab.update(multi_feats_cat_dict)
        self.morph_tag_cat_vocab["_"] = "_"

        # Assert that none of the feature-category mappings have been overwritten by accident
        assert all(
            self.morph_tag_cat_vocab[k] == uni_morph_tags[k][1]
            for k in uni_morph_tags.keys()
        ), "Feature to cat. from file overwritten"

        n_still_unmatched = (
            len([feat for feat, cat in self.morph_tag_cat_vocab.items() if cat == "_"])
            - 1
        )
        if n_still_unmatched >= 1:
            print(f"Number of features without category: {n_still_unmatched}")

        # Generate cat to int mapping
        self.morph_cat_vocab = {
            k: v for v, k in enumerate(sorted(set(self.morph_tag_cat_vocab.values())))
        }
        self.morph_cat_vocab["_"] = len(self.morph_cat_vocab) - 1

    def _move_to_pt(self):

        for d in self.docs:
            chars_tensor = [
                torch.tensor(
                    self.char_vocab.lookup_indices([c for c in t]), dtype=torch.long
                )
                for t in d.tokens
            ]

            tokens_tensor = torch.tensor(
                self.token_vocab.lookup_indices([t for t in d.tokens]), dtype=torch.long
            )

            morph_tags_tensor = torch.stack(
                [
                    torch.sum(
                        F.one_hot(
                            torch.tensor(
                                [self.morph_tag_vocab.get(tag, "_") for tag in tagset],
                                dtype=torch.long,
                            ),
                            len(self.morph_tag_vocab),
                        ),
                        dim=0,
                    )
                    for tagset in d.morph_tags
                ],
                dim=0,
            )

            morph_cats_tensor = torch.stack(
                [
                    torch.sum(
                        F.one_hot(
                            torch.tensor(
                                list(
                                    {
                                        self.morph_cat_vocab[
                                            self.morph_tag_cat_vocab[tag.lower()]
                                        ]
                                        for tag in tagset
                                    }
                                ),
                                dtype=torch.long,
                            ),
                            len(self.morph_cat_vocab),
                        ),
                        dim=0,
                    )
                    for tagset in d.morph_tags
                ],
                dim=0,
            )

            d.set_tensors(
                chars_tensor, tokens_tensor, morph_tags_tensor, morph_cats_tensor
            )

    def setup(self, generate_tensors: bool = True):
        """Code to run when finished importing all files.
        """

        print(f"CORPUS SETUP")
        # Iterate over the documents to get the necessary vocabs

        print(f"Splits logging")
        # Collect information about documents
        self.treebanks = defaultdict(int)
        self.splits = defaultdict(list)

        for i, doc in enumerate(self.docs):

            self.treebanks[f"{doc.treebank}_{doc.language}"] += 1
            self.splits[doc.split] += [i]

        if self.sorted:
            print(f"Length sorting")
            self.splits["train"] = sorted(
                self.splits["train"], key=lambda x: len(self.docs[x]), reverse=True
            )

            self.splits["dev"] = sorted(
                self.splits["dev"], key=lambda x: len(self.docs[x]), reverse=True
            )

            self.splits["test"] = sorted(
                self.splits["test"], key=lambda x: len(self.docs[x]), reverse=True
            )

        self.treebanks = dict(self.treebanks)
        self.splits = dict(self.splits)

        print(f"Generating lemma scripts")
        self._set_lemma_tags()

        print(f"Getting vocabs")
        self._get_vocabs()

        if generate_tensors:
            print(f"Generating tensors")
            # Move documents information to tensors for minibatching
            self._move_to_pt()

    def collate_batch(self, batch) -> Tuple[torch.Tensor]:

        docs_subset = [
            [
                d.chars_tensor,
                d.tokens,
                d.tokens_tensor,
                d.pretrained_embeddings,
                d.morph_tags_tensor,
                d.lemma_tags_tensor,
                d.morph_cats_tensor,
            ]
            for d in batch
        ]

        (
            chars,
            tokens_raw,
            tokens,
            pretrained_embeddings,
            morph_tags,
            lemma_tags,
            morph_cats,
        ) = list(map(list, zip(*docs_subset)))

        # Characters [T_c, B]
        char_lens = [c.size(0) for seq in chars for c in seq]

        chars = pad_sequence(
            [c for seq in chars for c in seq],
            padding_value=self.char_vocab[self.pad_token],
        )

        # Tokens [T_t, B]
        token_lens = [seq.size(0) for seq in tokens]

        tokens_raw = [[token for token in seq] for seq in tokens_raw]

        tokens = pad_sequence(tokens, padding_value=0)

        if self.pretrained_embeddings_dim != 0:
            pretrained_embeddings = pad_sequence(pretrained_embeddings, padding_value=0)
        else:
            pretrained_embeddings = None

        # Tags [T_t x B, C]/[T_t x B]
        lemma_tags = pad_sequence(
            lemma_tags, batch_first=self.batch_first, padding_value=-1
        )

        morph_tags = pad_sequence(
            morph_tags, batch_first=self.batch_first, padding_value=-1
        )

        morph_cats = pad_sequence(
            morph_cats, batch_first=self.batch_first, padding_value=-1
        )

        if self.return_tokens_raw:
            return (
                char_lens,
                chars,
                token_lens,
                tokens_raw,
                tokens,
                pretrained_embeddings,
                lemma_tags,
                morph_tags,
                morph_cats,
            )

        else:
            return (
                char_lens,
                chars,
                token_lens,
                tokens,
                pretrained_embeddings,
                lemma_tags,
                morph_tags,
                morph_cats,
            )

    @property
    def pretrained_embeddings_dim(self) -> int:

        if len(self.docs) == 0:
            raise ValueError("This corpus has no documents.")

        else:
            try:
                if isinstance(self.docs[0].pretrained_embeddings, torch.Tensor):
                    return self.docs[0].pretrained_embeddings.size(-1)
                elif self.docs[0].pretrained_embeddings == 0:
                    return 0
            except ValueError:
                return 0

    def lemma_tags_overview(self, n: int = 11) -> pd.DataFrame:
        """Get the most common lemma scripts and some examples.

        Args:
            n (int, optional): number of scripts to show. Defaults to 11.
        """

        most_common_rules = [
            [script, count] for script, count in self.script_counter.most_common(n)
        ]

        for entry in most_common_rules:
            entry.append(self.script_examples[entry[0]])

        df = pd.DataFrame(most_common_rules, columns=["Rule", "Count", "Examples"])

        return df


class TreebankDataModule(pl.LightningDataModule):
    """A PyTorch-Lightning datamodule that regulates the construction, loading, saving and batching of Treebank Corpora.

    Args:
        batch_size (int): the default batchsize. Can be overwritten when calling individual dataloaders
        language (Optional[str], optional): the treebank's language(s) (see `DocumentCorpus'). Defaults to None.
        treebank_name (Optional[str], optional): the treebank's name (see `DocumentCorpus'). Defaults to None.
        batch_first (Optional[bool], optional): whether tensor's first dimension is the batch size or sequence length. Defaults to True.
        len_sorted (Optional[bool], optional): whether or not to length sort the dataset. Defaults to True.
        unk_token (Optional[str], optional): defaults to "<UNK>".
        pad_token (Optional[str], optional): defaults to "<PAD>".
        return_tokens_raw (Optional[bool], optional): see `DocumentCorpus'. Defaults to True.
        max_tokens (int, optional): max number of tokens per document (note, this is raw tokens, not BPE/SPE). Defaults to 256.
        max_chars (int, optional): max number of chars per document. Defaults to 2048.
        remove_duplicates (bool, optional): remove duplicate texts. Defaults to True.
        remove_unique_lemma_scripts (bool, optional): removes lemma scripts with only a single lemma (not recommended). Defaults to False.
        include_family (bool, optional): includes typological family when searching for CONLLU files (i.e. all Germanic languages). Defaults to False.
        family_level (str, optional): which level to group languages by, `sibling' indicates same family (Dutch -> Germanic), `parent' indicates same genus (Dutch -> Indo-European). Defaults to "sibling".
        quality_limit (float, optional): lower limit of UD quality estimate. See site for more details. Very low values indicate low quality annotations. Defaults to 0.0.
        source (Optional[str], optional): directory in which files CONLLU files can be found. Defaults to None.
    """

    def __init__(
        self,
        batch_size: int,
        language: Optional[Union[str, List[str]]] = None,
        treebank_name: Optional[str] = None,
        batch_first: Optional[bool] = True,
        len_sorted: Optional[bool] = True,
        unk_token: Optional[str] = "<UNK>",
        pad_token: Optional[str] = "<PAD>",
        return_tokens_raw: Optional[bool] = True,
        max_tokens: int = 256,
        max_chars: int = 2048,
        remove_duplicates: bool = True,
        remove_unique_lemma_scripts: bool = False,
        include_family: bool = False,
        family_level: str = "sibling",
        quality_limit: float = 0.0,
        source: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(language, str):
            self.language = [language]
        elif isinstance(language, list) or isinstance(
            language, omegaconf.listconfig.ListConfig
        ):
            self.language = [*language]
        else:
            raise ValueError(f"{type(language)} is not supported input for language.")

        self.treebank_name = treebank_name
        self.batch_first = batch_first
        self.len_sorted = len_sorted
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.return_tokens_raw = return_tokens_raw
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.remove_duplicates = remove_duplicates
        self.remove_unique_lemma_scripts = remove_unique_lemma_scripts
        self.batch_size = batch_size
        self.include_family = include_family
        self.family_level = family_level
        self.quality_limit = quality_limit
        self.source = source if isinstance(source, str) else BASEPATH

    def prepare_data(self, **kwargs) -> None:

        print("FINDING CONNLU FILES")
        print(f"Looking at:\n\t{Path(self.source).absolute()}")

        if self.include_family:
            print("Extending with linguistic family.")

            included_families = {LANGUAGE_TO_FAMILY[lang] for lang in self.language}

            if "parent" in self.family_level.lower():
                # If include families is set to the parent level, includes all sub-types of the typological family
                # For example, if Finnish is an included language, which if of type (Uralic, Finnic)
                # Hungarian, of type (Uralic, Hungarian) is also included
                included_families = [
                    k
                    for fam in included_families
                    for k in FAMILY_TO_LANGUAGE.keys()
                    if fam.split(", ")[0] in k
                ]

            extended_languages = [
                lang
                for family in included_families
                for lang in FAMILY_TO_LANGUAGE[family]
            ]

            files = [
                f
                for lang in extended_languages
                for f in get_conllu_files(
                    lang, name=self.treebank_name, source=self.source
                )
            ]

            self._included_languages = extended_languages

            print(
                f"Languages included: {len(self.language)}->{len(self._included_languages)}"
            )

        else:
            files = [
                f
                for lang in self.language
                for f in get_conllu_files(
                    lang, name=self.treebank_name, source=self.source
                )
            ]

            self._included_languages = self.language

        if self.quality_limit > 0.0:
            filtered_files = []

            quality_limit = 0.2

            n_kept, n_removed = 0, 0
            n_sentences_kept, n_sentences_removed = 0, 0
            for f in files:

                _, tb, _, lang = f

                tb_metadata = TREEBANK_METADATA[f"{lang}_{tb}"]

                if tb_metadata["quality"] > quality_limit:
                    filtered_files.append(f)

                    n_kept += 1
                    n_sentences_kept += tb_metadata["size"]["sentences"]

                else:
                    n_removed += 1
                    n_sentences_removed += tb_metadata["size"]["sentences"]

            n_kept = n_kept // 3
            n_removed = n_removed // 3
            n_sentences_kept = n_sentences_kept // 3
            n_sentences_removed = n_sentences_removed // 3

            print(
                f"Corpora included: {n_kept}/{n_kept + n_removed}, {n_kept / (n_kept + n_removed) * 100:.2f}%"
            )
            print(
                f"Sentences included: {n_sentences_kept}/{n_sentences_kept + n_sentences_removed}, "
                + f"{n_sentences_kept / (n_sentences_kept + n_sentences_removed) * 100:.2f}%"
            )

            files = filtered_files

        self.corpus = DocumentCorpus(
            batch_first=self.batch_first,
            sorted=self.len_sorted,
            return_tokens_raw=self.return_tokens_raw,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            max_tokens=self.max_tokens,
            max_chars=self.max_chars,
            remove_unique_lemma_scripts=self.remove_unique_lemma_scripts,
        )

        for i, (fp, name, split, language) in enumerate(files):
            self.corpus.parse_connlu_file(
                fp, split, name, language, remove_duplicates=self.remove_duplicates
            )

        self.corpus.setup(**kwargs)

        print()
        print(self.corpus)

    def setup(self, stage: Optional[str] = None):

        if stage in (None, "fit"):
            self.train_corpus = Subset(self.corpus, self.corpus.splits["train"])

            self.valid_corpus = Subset(self.corpus, self.corpus.splits["dev"])

        if stage in (None, "test"):
            self.test_corpus = Subset(self.corpus, self.corpus.splits["test"])

    def _get_dataloader(
        self,
        split: str,
        batch_size: Optional[int] = None,
        use_pytorch: bool = True,
        **dataloader_kwargs,
    ):
        if split == "train":
            corpus = self.train_corpus
        elif split == "val" or split == "valid":
            corpus = self.valid_corpus
        elif split == "test":
            corpus = self.test_corpus

        if batch_size is None:
            batch_size = self.batch_size

        if use_pytorch:
            data_loader = DataLoader(
                corpus,
                batch_size=batch_size,
                shuffle=False if self.len_sorted else True,
                collate_fn=self.corpus.collate_batch,
                **dataloader_kwargs,
            )
        else:
            data_loader = TokenDataloader(
                dataset=corpus,
                collate_fn=self.corpus.collate_batch,
                **dataloader_kwargs,
            )

        return data_loader

    def train_dataloader(
        self,
        batch_size: Optional[int] = None,
        use_pytorch: bool = True,
        **dataloader_kwargs,
    ):
        return self._get_dataloader(
            "train", batch_size, use_pytorch, **dataloader_kwargs
        )

    def val_dataloader(
        self,
        batch_size: Optional[int] = None,
        use_pytorch: bool = True,
        **dataloader_kwargs,
    ):
        return self._get_dataloader("val", batch_size, use_pytorch, **dataloader_kwargs)

    def test_dataloader(
        self,
        batch_size: Optional[int] = None,
        use_pytorch: bool = True,
        **dataloader_kwargs,
    ):
        return self._get_dataloader(
            "test", batch_size, use_pytorch, **dataloader_kwargs
        )

    def save(self, fp):
        with open(fp, "wb") as fp:
            pickle.dump(self, fp)

    @classmethod
    def load(cls, fp):

        with open(fp, "rb") as fp:
            return pickle.load(fp)
