from typing import Any
from functools import partial

from sacremoses import MosesTokenizer

class MosesTokenizerWrapped(object):
    """Wrapper for a MosesTokenizer.

    Args:
        object (_type_): _description_
    """

    def __init__(self, lang):
        self.tokenizer = partial(MosesTokenizer(lang=lang).tokenize, escape=False)

    def __call__(self, texts) -> Any:

        if isinstance(texts, str):
            return self.tokenizer(texts)
        if isinstance(texts, list) and isinstance(texts[0], str):
            return [self.tokenizer(txt) for txt in texts]