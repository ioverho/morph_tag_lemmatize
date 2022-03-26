import os
import json
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import re
from collections import defaultdict

from bs4 import BeautifulSoup, Comment, Tag

URL = "https://universaldependencies.org/"


def scrape():
    """Scrape the UD project page for metadata of the treebanks.
    """
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.get(URL)

    webpage_content = session.get(URL).text

    soup = BeautifulSoup(webpage_content, "html.parser")

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    treebank_metadata = {}
    language_to_family = {}
    for c in comments:

        # Get matadata for specific treebanks (size, quality)
        matches = re.search(r"(?<=start of )(.*) \/ (.*)(?= entry)", c.string)
        if matches is not None:
            language, treebank = matches.group(1), matches.group(2)
            language = language.replace(" ", "_")
            parent = c.parent

            meta_data = []
            for i, subtag in enumerate(parent.contents):
                if isinstance(subtag, Tag):
                    if len(subtag.contents):
                        if isinstance(subtag.contents[0], Tag):
                            subsubstag = subtag.contents[0]
                        else:
                            continue

                        meta_data.append(subsubstag.attrs.get("data-hint"))

            size_matches = re.search(
                r"(.+)(?: tokens )(.+)(?: words )(.+)(?: sentences)", meta_data[0]
            )

            treebank_metadata[f"{language}_{treebank}"] = {
                "language": language,
                "size": {
                    "tokens": int(size_matches.group(1).replace(",", "")),
                    "words": int(size_matches.group(2).replace(",", "")),
                    "sentences": int(size_matches.group(3).replace(",", "")),
                },
                "source_genres": meta_data[2].split(" "),
                "quality": float(meta_data[4]),
                "license": meta_data[3],
            }

        # Get metadata on languages and their families
        matches = re.search("(?<=start of )(.*)(?= accordion row)", c.string)
        if matches is not None:
            language = matches.group(1)
            typological_information = c.parent.contents[-2].contents[0]

            language_to_family[language] = typological_information

    # Hardcoded name changes
    treebank_metadata["French_Spoken"] = treebank_metadata["French_Rhapsodie"]

    treebank_metadata["Polish_SZ"] = treebank_metadata["Polish_PDB"]

    # Add inverse language to lingusitic-family mapping
    family_to_language = defaultdict(list)
    for k, v in language_to_family.items():
        family_to_language[v].append(k)
    family_to_language = dict(family_to_language)

    return treebank_metadata, language_to_family, family_to_language


if __name__ == "__main__":

    treebank_metadata, language_to_family, family_to_language = scrape()

    with open("./morphological_tagging/data/treebank_metadata.json", "w") as f1:
        json.dump(treebank_metadata, f1)

    with open("./morphological_tagging/data/language_to_family.json", "w") as f2:
        json.dump(language_to_family, f2)

    with open("./morphological_tagging/data/family_to_language.json", "w") as f3:
        json.dump(family_to_language, f3)
