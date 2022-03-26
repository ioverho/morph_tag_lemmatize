import os
import pickle
from pathlib import Path
import re

import hydra
import torch
import numpy as np
import nltk
from nltk.metrics.distance import edit_distance

from utils.experiment import progressbar, Timer
from morphological_tagging.data.lemma_script import apply_lemma_script
from morphological_tagging.data.corpus import TreebankDataModule
from morphological_tagging.pipelines import UDPipe2Pipeline
from morphological_tagging.metrics import RunningStats

EVAL_PATH = Path("./morphological_tagging/eval")
CORPORA_PATH = Path("./morphological_tagging/data/corpora")
CHECKPOINT_PATH = Path("./morphological_tagging/checkpoints")

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="./morphological_tagging/config", config_name="eval")
def eval(config):

    timer = Timer()

    print(f"\n{timer.time()} | INITIALIZATION")
    use_cuda = config["gpu"] or config["gpu"] > 1
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Running on {device}" + f"- {torch.cuda.get_device_name(0)}"
        if use_cuda
        else ""
    )

    # ==========================================================================
    # Dataset Import
    # ==========================================================================
    print(f"\n{timer.time()} | DATASET")

    expected_dataset_path = (
        CORPORA_PATH
        / f"{config['dataset_name']}_{config['quality_limit']}_{config['batch_first']}.pickle"
    )

    if expected_dataset_path.exists():
        print(f"Found predefined dataset at {expected_dataset_path}")

    data_module = TreebankDataModule.load(expected_dataset_path)

    # ==========================================================================
    # Model Import
    # ==========================================================================
    print(f"\n{timer.time()} | MODEL")
    chkpt_dir_path = f"**/{config['model_name']}_{config['dataset_name']}/*"

    matches = list(CHECKPOINT_PATH.glob(chkpt_dir_path))

    if len(matches) == 0:
        raise ValueError(
            f"No model checkpoint found at {CHECKPOINT_PATH}/{config['model_name']}_{config['dataset_name']}"
        )

    elif len(matches) >= 1:
        dirs = []
        for m in matches:
            if len(list(m.glob("**/*.ckpt"))) > 0:
                re_search = re.search("(?<=version_)([0-9]+)", m.parts[-1])
                dirs.append((m, int(re_search.groups()[0])))

        latest_dir = sorted(dirs, key=lambda x: x[1], reverse=True)[0][0]

        if config["mode"] == "best":
            match = list((latest_dir / "checkpoints").glob("**/epoch*.ckpt"))[0]

        elif config["mode"] == "last":
            match = list((latest_dir / "checkpoints").glob("**/last.ckpt"))[0]

        else:
            raise ValueError(f"Mode {config['mode']} not recognized.")

        print(f"Found model at {match}")

    if config["model_name"].lower() == "udpipe2":
        pipeline = UDPipe2Pipeline()

        pipeline.load_vocabs_from_treebankdatamodule_checkpoint(expected_dataset_path)
        pipeline.load_tagger(str(match), map_location=device)

        pipeline = pipeline.to(device)

    elif config["model_name"].lower() == "udify":
        raise NotImplementedError()

    # ==========================================================================
    # Eval File
    # ==========================================================================
    print(f"\n{timer.time()} | EVALUATION")
    eval_fp = (
        EVAL_PATH
        / f"{config['model_name']}_{config['dataset_name']}_{config['quality_limit']}_{config['batch_first']}.pickle"
    )
    print(f"Saving output to {eval_fp}")
    if EVAL_PATH.exists() and config["overwrite_ok"]:
        print(">>>OVERWRITING<<<")

    data_loader = data_module.test_dataloader(
        max_tokens=config["max_tokens"],
        max_batch_size=config["max_batch_size"],
        use_pytorch=False,
    )

    # ==============================================================================
    # Evaluation loop
    # ==============================================================================
    lemma_acc = RunningStats()
    lev_dist = RunningStats()
    morph_set_acc = RunningStats()
    morph_set_iou = RunningStats()
    morph_true_pos = np.zeros((len(pipeline.id_to_morph_tag),))
    morph_pos = np.zeros((len(pipeline.id_to_morph_tag),))
    morph_rel = np.zeros((len(pipeline.id_to_morph_tag),))

    n_tokens = 0
    timer2 = Timer(silent=True)
    with open(eval_fp, "wb") as f:
        for batch in progressbar(data_loader, "Evaluating", size=120):

            tokens = batch[3]
            for tok_seq in tokens:
                n_tokens += len(tok_seq)

            ls_preds_ids, mt_preds_ = pipeline.predict(
                batch[0], batch[1], batch[2], batch[3], batch[4]
            )
            lemmas, lemma_scripts, morph_tags, morph_cats = pipeline.preds_to_text(
                tokens, ls_preds_ids, mt_preds_
            )

            for i in range(len(tokens)):
                assert (
                    len(tokens[i])
                    == len(lemmas[i])
                    == len(lemma_scripts[i])
                    == len(morph_tags[i])
                    == len(morph_cats[i])
                )

            ls_preds_ids = ls_preds_ids.detach().cpu().permute(1, 0).numpy()
            mt_preds_ = mt_preds_.detach().cpu().permute(1, 0, 2).numpy()

            ls_gts = [
                [pipeline.id_to_lemma_script[ls] for ls in ls_seq]
                for ls_seq in batch[6].permute(1, 0).numpy()
            ]

            mt_gts_ = batch[7].permute(1, 0, 2).numpy()
            mt_gts = [
                [
                    set(pipeline.id_to_morph_tag[mt] for mt in np.where(mt_gt)[0])
                    for mt_gt in mt_gt_seq
                    if -1 not in mt_gt
                ]
                for mt_gt_seq in mt_gts_
            ]

            for (
                tok_seq,
                lm_pred_seq,
                ls_pred_seq,
                ls_gt_seq,
                mt_pred_seq,
                mt_gt_seq,
                mt_pred_seq_,
                mt_gt_seq_,
            ) in zip(
                tokens,
                lemmas,
                lemma_scripts,
                ls_gts,
                morph_tags,
                mt_gts,
                mt_preds_,
                mt_gts_,
            ):
                for (
                    token,
                    lm_pred,
                    ls_pred,
                    ls_gt,
                    mt_pred,
                    mt_gt,
                    mt_pred_,
                    mt_gt_,
                ) in zip(
                    tok_seq,
                    lm_pred_seq,
                    ls_pred_seq,
                    ls_gt_seq,
                    mt_pred_seq,
                    mt_gt_seq,
                    mt_pred_seq_,
                    mt_gt_seq_,
                ):
                    lm_gt = apply_lemma_script(token, ls_gt, verbose=False)

                    lemma_acc(lm_pred == lm_gt)
                    lev_dist(edit_distance(lm_pred, lm_gt))

                    inter = set.intersection(mt_pred, mt_gt)
                    union = set.union(mt_pred, mt_gt)
                    morph_set_acc(mt_pred == mt_gt)
                    morph_set_iou(len(inter) / len(union))

                    morph_true_pos += np.logical_and(mt_pred_, mt_gt_)
                    morph_pos += mt_pred_
                    morph_rel += mt_gt_

                    pickle.dump(
                        (token, lm_gt, lm_pred, ls_gt, ls_pred, mt_gt, mt_pred,), f,
                    )

        token_rate = n_tokens / timer2.time().seconds

        # ==============================================================================
        # Generate stats
        # ==============================================================================
        morph_prec = morph_true_pos / np.clip(morph_pos, 1, None)
        morph_recall = morph_true_pos / np.clip(morph_rel, 1, None)
        morph_f1 = (
            2
            * (morph_prec * morph_recall)
            / np.clip((morph_prec + morph_recall), 1, None)
        )
        pres_idx = np.where(morph_rel > 0)[
            0
        ]  # Dealing with morph tags that are not present in the test set
        morph_f1_micro = np.sum(morph_rel * morph_f1) / np.sum(morph_rel)
        morph_f1_macro = np.mean(morph_f1[pres_idx])
        f1_per_tag = sorted(
            [
                (
                    pipeline.id_to_morph_tag[i],
                    morph_f1[i],
                    int(morph_rel[i]),
                    morph_prec[i],
                    morph_recall[i],
                )
                for i in pres_idx
            ],
            key=lambda x: (x[1], x[2]),
        )

        pipeline.performance_stats[
            "lemma_acc"
        ] = f"{lemma_acc.mean:.2f} +- {lemma_acc.se:.2e}"
        pipeline.performance_stats[
            "lev_dist"
        ] = f"{lev_dist.mean:.2f} +- {lev_dist.se:.2e}"
        pipeline.performance_stats[
            "morph_set_acc"
        ] = f"{morph_set_acc.mean:.2f} +- {morph_set_acc.se:.2e}"
        pipeline.performance_stats["morph_tag_f1_micro"] = f"{morph_f1_micro:.2f}"
        pipeline.performance_stats["morph_tag_f1_macro"] = f"{morph_f1_macro:.2f}"
        pipeline.performance_stats["morph_tag_f1"] = [
            (tag, f"F1: {f1:.2f} ({prec:.2f}/{recl:.2f}), N: {N:d}")
            for tag, f1, N, prec, recl in f1_per_tag
        ]
        pipeline.performance_stats["tokens_per_second"] = f"{token_rate:.2f}"

        pickle.dump(pipeline.performance_stats, f)

    # ==============================================================================
    # Wrap up and save pipeline
    # ==============================================================================
    for k, v in pipeline.performance_stats.items():
        if k == "morph_tag_f1":
            print(k)
            for kk, vv in pipeline.performance_stats[k]:
                print(f"\t{kk}: {vv}")
        else:
            print(f"{k}: {v}")

    pipeline = pipeline.to("cpu")
    pipeline.save(
        f"./morphological_tagging/pipelines/{config['model_name']}_{config['dataset_name']}.ckpt"
    )

    print(f"\n{timer.time()} | FINISHED")

    return eval_fp


if __name__ == "__main__":

    eval_fp = eval()
