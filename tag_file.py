import os
from pathlib import Path
import argparse
import math
from collections import defaultdict
import pickle

import torch

from morphological_tagging.pipelines import UDPipe2Pipeline, DogTagPipeline
from utils.tokenizers import MosesTokenizerWrapped
from utils.experiment import Timer, progressbar

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

CORPORA_LOC = "./nmt_adapt/data/corpora/"

def tag_file(args):

    for format in args.output_format:
        assert  format in {"single_pickle_file", "separate_text_files", "single_jsonlines_file"}, f"Output format {format} not recognized."

    timer = Timer()

    print(f"\n{timer.time()} | SETUP")
    # == Device
    device = torch.device("cuda" if args.gpu > 0 else "cpu")
    print(
        f"Training on {device}" + (f"- {torch.cuda.get_device_name(0)}"
        if args.gpu
        else "")
    )

    print(f"\n{timer.time()} | MODEL IMPORT")
    expected_pipeline_path = f"./morphological_tagging/pipelines/{args.pipeline}_{args.language}_merge.ckpt"
    print(f"Looking for pipeline in {expected_pipeline_path}")

    if "udpipe" in args.pipeline.lower():
        pipeline = UDPipe2Pipeline.load(expected_pipeline_path)

    elif "dogtag" in args.pipeline.lower():
        pipeline = DogTagPipeline.load(expected_pipeline_path)

    else:
        raise ValueError(f"Architecture cannot be inferred from {args.pipeline}. 'udpipe' or 'dogtag' must be included.")

    pipeline.tagger.eval()
    for param in pipeline.parameters():
        param.requires_grad = False
    pipeline = pipeline.to(device)

    pipeline.add_tokenizer(MosesTokenizerWrapped(lang=args.language))

    # Import files
    lines = []
    with open(args.file_path, encoding="utf-8" if args.encoding is None else args.encoding) as f:
        lines.extend([line.strip() for line in f.readlines()])

    # Tagging
    print(f"\n{timer.time()} | TAGGING")
    tag_results = defaultdict(list)

    i = 0
    for i in progressbar(
        range(math.ceil(len(lines) / args.batch_size)),
        prefix="Tagging"
        ):

        batch = lines[i*args.batch_size:(i+1)*args.batch_size]

        lemmas, lemma_scripts, morph_tags, morph_cats = pipeline.forward(
                batch,
                is_pre_tokenized=False,
                )

        tag_results["lemmas"].extend(lemmas)
        tag_results["lemma_scripts"].extend(lemma_scripts)
        tag_results["morph_tags"].extend(morph_tags)
        tag_results["morph_cats"].extend(morph_cats)

    tag_results = dict(tag_results)
    tag_results["morph_tags"] = [
        [";".join(sorted(list(mt_set))) for mt_set in seq]
        for seq in tag_results["morph_tags"]
    ]

    tag_results["morph_cats"] = [
        [";".join(sorted(list(mt_set))) for mt_set in seq]
        for seq in tag_results["morph_cats"]
    ]

    print(f"\n{timer.time()} | WRITING FILE(S)")

    for format in args.output_format:
        if format == "single_pickle_file":
            out_fp = Path(args.file_path).parent / (Path(args.file_path).stem + "_tagged.pickle")
            print(f"Saving to: {out_fp}")
            with open(out_fp, "wb") as f:
                pickle.dump(tag_results, file=f)

        if format == "separate_text_files":

            for k in tag_results.keys():
                out_fp = Path(args.file_path).parent / (Path(args.file_path).stem + f"_{k}.txt")
                print(f"Saving to: {out_fp}")
                lines_ = [" ".join(seq) + "\n" for seq in tag_results[k]]
                with open(out_fp, "w", encoding="utf-8" if args.encoding is None else args.encoding) as f:
                    f.writelines(lines_)

        if format == "single_jsonlines_file":

            import jsonlines

            tag_results_ = [
                {
                    "lemma": lemma,
                    "lemma_script": lemma_script,
                    "morph_tag": morph_tag,
                    "morph_cat": morph_cat
                    } for lemma, lemma_script, morph_tag, morph_cat in zip(
                    tag_results["lemmas"],
                    tag_results["lemma_scripts"],
                    tag_results["morph_tags"],
                    tag_results["morph_cats"])
                    ]

            out_fp = Path(args.file_path).parent / (Path(args.file_path).stem + f"_tagged.jsonl")
            print(f"Saving to: {out_fp}")
            with jsonlines.open(out_fp, mode='w') as writer:
                writer.write(tag_results_)

    timer.end()

    return 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Annotate a text file with morphological tags and lemmas.'
        )

    parser.add_argument(
        '--file_path',
        type=str,
        help='the location of the text file'
        )
    parser.add_argument(
        '--language',
        type=str,
        help='the language of the text'
        )
    parser.add_argument(
        '--pipeline',
        default="UDPipe2",
        type=str,
        help="pipeline checkpoint name in './pipelines'"
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=1,
        help='whether to annotate on GPU, if available'
        )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='number of lines being fed into the pipeline'
        )
    parser.add_argument(
        '--encoding',
        type=str,
        help="encoding of text file"
        )
    parser.add_argument(
        '--output_format',
        type=str,
        nargs="+",
        choices=["single_pickle_file", "separate_text_files", "single_jsonlines_file"],
        default=["separate_text_files"],
        help="output format"
        )

    args = parser.parse_args()

    tag_file(args)
