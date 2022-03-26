# %%
from pathlib import Path
import json
from collections import defaultdict

import pandas as pd

with open("./morphological_tagging/data/treebank_metadata.json") as f:
    TREEBANK_METADATA = json.load(f)

with open("./morphological_tagging/data/family_to_language.json") as f:
    FAMILY_TO_LANGUAGE = json.load(f)

with open("./morphological_tagging/data/language_to_family.json") as f:
    LANGUAGE_TO_FAMILY = json.load(f)

with open("./morphological_tagging/data/supported_languages.json") as f:
    SUPPORTED_LANGUAGES = json.load(f)

data_dir = "./morphological_tagging/data/um-treebanks-v2.9"

table_data = defaultdict(dict)
for fp in Path(data_dir).glob("**"):
    if not (fp.is_dir() and "UD_" in fp.parts[-1]):
        continue

    name_lang = fp.parts[-1].split("_", maxsplit=1)[-1]
    lang, name = name_lang.split("-")
    meta_data = TREEBANK_METADATA[name_lang.replace("-", "_")]

    lang = lang.replace("_", " ")

    table_data[(lang, name)][("Size", "Sentences")] = meta_data["size"]["sentences"]
    table_data[(lang, name)][("Size", "Tokens")] = meta_data["size"]["tokens"]

    table_data[(lang, name)][("Sources", "Genres")] = meta_data["source_genres"]
    table_data[(lang, name)][("Sources", "N_Genres")] = len(meta_data["source_genres"])

    table_data[(lang, name)][("Quality", "Rating")] = meta_data["quality"]
    table_data[(lang, name)][
        ("Quality", "Stars")
    ] = f"{round(meta_data['quality']*5 *2)/2:3.1f}"

    table_data[(lang, name)][("Support", "FastText")] = (
        lang in SUPPORTED_LANGUAGES["fasttext"]
    )
    table_data[(lang, name)][("Support", "BERT")] = lang in SUPPORTED_LANGUAGES["bert"]
    table_data[(lang, name)][("Support", "XLM-R")] = (
        lang in SUPPORTED_LANGUAGES["roberta"]
    )

df = pd.DataFrame.from_dict(table_data, orient="index")
df = df.rename_axis(["Language", "Treebank"])

df

# %%

stars_limit = 0
necessary_support = ["FastText", "BERT"]  # "XLM-R"

langs_summary = defaultdict(dict)
for index, series in df.iterrows():
    lang, _ = index

    if float(series[("Quality", "Stars")]) < stars_limit:
        continue

    flag = False
    for pretrained_module in necessary_support:
        if not series[("Support", pretrained_module)]:
            flag = True
    if flag:
        continue

    cur_stats = langs_summary[lang]

    cur_stats[("Typology", "Family")] = LANGUAGE_TO_FAMILY[lang]

    cur_stats[("General", "N_Treebanks")] = (
        cur_stats.get(("General", "N_Treebanks"), 0) + 1
    )

    cur_stats[("Size", "Sentences")] = (
        cur_stats.get(("Size", "Sentences"), 0) + series[("Size", "Sentences")]
    )
    cur_stats[("Size", "Tokens")] = (
        cur_stats.get(("Size", "Tokens"), 0) + series[("Size", "Tokens")]
    )

    cur_stats[("Sources", "Genres")] = cur_stats.get(
        ("Sources", "Genres"), set()
    ) | set(series[("Sources", "Genres")])
    cur_stats[("Sources", "N_Genres")] = len(
        cur_stats.get(("Sources", "Genres"), set())
    )

    cur_stats[("Quality", "Rating")] = (
        cur_stats.get(("Quality", "Rating"), 0)
        + (series[("Quality", "Rating")] - cur_stats.get(("Quality", "Rating"), 0))
        / cur_stats[("General", "N_Treebanks")]
    )
    cur_stats[
        ("Quality", "Stars")
    ] = f"{round(cur_stats[('Quality', 'Rating')]*5 *2)/2:3.1f}"

    cur_stats[("Support", "FastText")] = lang in SUPPORTED_LANGUAGES["fasttext"]
    cur_stats[("Support", "BERT")] = lang in SUPPORTED_LANGUAGES["bert"]
    cur_stats[("Support", "XLM-R")] = lang in SUPPORTED_LANGUAGES["roberta"]

df2 = pd.DataFrame.from_dict(langs_summary, orient="index")

df2

# %%

df2.to_clipboard()