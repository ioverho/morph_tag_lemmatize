import os

import hydra
from omegaconf import DictConfig

from morphological_tagging.data.corpus import TreebankDataModule

CHECKPOINT_DIR = "morphological_tagging/data/corpora"


@hydra.main(config_path="./morphological_tagging/config", config_name="treebank_corpus")
def build(config: DictConfig):
    """Builds a treebank data module from provide parameters, and saves to disk
    """

    print(f"Language: {config['language']}")
    print(f"Treebanks: {config['treebank_name']}")
    print(f"Include linguistic family: {config['include_family']}")
    print(f"Family level: {config['family_level']}")
    print(f"Quality limit: {config['quality_limit']}")
    print(f"Batch first: {config['batch_first']}\n")

    data_module = TreebankDataModule(
        batch_size=config["batch_size"],
        language=config["language"],
        treebank_name=config["treebank_name"],
        batch_first=config["batch_first"],
        remove_unique_lemma_scripts=config["remove_unique_lemma_scripts"],
        include_family=config["include_family"],
        family_level=config["family_level"],
        quality_limit=config["quality_limit"],
        return_tokens_raw=config["return_tokens_raw"],
        len_sorted=config["len_sorted"],
        max_chars=config["max_chars"],
        max_tokens=config["max_tokens"],
        remove_duplicates=config["remove_duplicates"],
        source=config["source"],
    )

    data_module.prepare_data()
    data_module.setup()

    if config["file_name"] is None:
        n_langs = len(data_module._included_languages)
        lang_id = config["language"] if n_langs == 1 else f"multi_{n_langs}"

        output_path = os.path.join(
            os.getcwd(),
            CHECKPOINT_DIR,
            f"{lang_id}_{config['treebank_name']}_{config['quality_limit']}_{config['batch_first']}.pickle",
        )

    else:
        output_path = os.path.join(os.getcwd(), CHECKPOINT_DIR, config["file_name"])

    print(f"Saving to: {output_path}")
    data_module.save(output_path)


if __name__ == "__main__":

    build()
