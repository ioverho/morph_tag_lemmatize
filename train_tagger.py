import os
import warnings
import yaml

# 3rd Party
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


# User-defined
from morphological_tagging.data.corpus import TreebankDataModule
from morphological_tagging.models import (
    UDIFYFineTune,
    UDPipe2,
    UDIFY,
    UDIFYFineTune,
    DogTag,
    DogTagSmall,
)
from utils.experiment import find_version, set_seed, set_deterministic, Timer
from utils.errors import ConfigurationError

CHECKPOINT_DIR = "./morphological_tagging/checkpoints"

dotenv.load_dotenv(override=True)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


@hydra.main(config_path="./morphological_tagging/config")
def train(config: DictConfig):
    """Train loop.

    """

    timer = Timer()

    # *==========================================================================
    # *Config reading
    # *==========================================================================
    if config["print_hparams"]:
        print(50 * "+")
        print(f"\n{timer.time()} | HYPER-PARAMETERS")
        print(OmegaConf.to_yaml(config))
        print(50 * "+")
    else:
        print(f"\n{timer.time()} | HYPER-PARAMETERS")
        print("Loaded.")

    # *==========================================================================
    # *Experiment
    # *==========================================================================
    print(f"\n{timer.time()} | EXPERIMENT SETUP")

    full_name = f"{config['experiment_name']}_{config['data']['language']}_{config['data']['treebank_name']}"

    full_version, experiment_dir, version = find_version(
        full_name, CHECKPOINT_DIR, debug=config["debug"]
    )

    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}", exist_ok=True)
    os.makedirs(f"{CHECKPOINT_DIR}/{full_version}/checkpoints", exist_ok=True)

    with open(f"{CHECKPOINT_DIR}/{full_version}/config.yaml", "w") as outfile:
        yaml.dump(OmegaConf.to_container(config, resolve=True), outfile)

    # == Device
    use_cuda = config["gpu"] or config["gpu"] > 1
    device = torch.device("cuda" if use_cuda else "cpu")
    print(
        f"Training on {device}" + f"- {torch.cuda.get_device_name(0)}"
        if use_cuda
        else ""
    )

    # == Reproducibility
    set_seed(config["seed"])
    if config["deterministic"]:
        set_deterministic()

    # == Logging
    print(f"\n{timer.time()} | LOGGER SETUP")
    if config["logging"]["logger"].lower() == "tensorboard":
        # ==== ./checkpoints/data_version/version_number

        print(f"Saving to {CHECKPOINT_DIR}/{full_version}")

        # os.path.join(save_dir, name, version)
        logger = TensorBoardLogger(
            save_dir=f"{CHECKPOINT_DIR}",
            name=f"{experiment_dir}",
            version=f"version_{version}",
            # **config["logging"]["logger_kwargs"],
        )

    elif config["logging"]["logger"].lower() in ["wandb", "weightsandbiases"]:

        logger = WandbLogger(
            save_dir=f"{CHECKPOINT_DIR}/{full_version}",
            group=config["experiment_name"],
            config=OmegaConf.to_container(config),
            settings=wandb.Settings(start_method="fork"),
            **config["logging"]["logger_kwargs"],
        )

    else:
        raise ConfigurationError("Logger not recognized.")

    # *==========================================================================
    # * Callbacks
    # *==========================================================================

    callbacks = []

    if config.get("save_checkpoints", True):
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{CHECKPOINT_DIR}/{full_version}/checkpoints",
            save_top_k=config["save_top_k"],
            monitor=config["monitor"],
            mode=config["monitor_mode"],
            auto_insert_metric_name=True,
            save_last=True,
            save_weights_only=True,
        )
        callbacks += [checkpoint_callback]

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks += [lr_monitor]

    device_monitor = DeviceStatsMonitor()
    callbacks += [device_monitor]

    prog_bar = TQDMProgressBar(refresh_rate=config["prog_bar_refresh_rate"])
    callbacks += [prog_bar]

    # *==========================================================================
    # * Dataset
    # *==========================================================================
    print(f"\n{timer.time()} | DATA SETUP")
    if config["data"].get("file_path", None) is None:
        data_module = TreebankDataModule(**config["data"])
        data_module.prepare_data()
        data_module.setup()
    elif os.path.exists(config["data"]["file_path"]):
        print(f"Loading pre-defined data module from {config['data']['file_path']}")
        data_module = TreebankDataModule.load(config["data"]["file_path"])
        data_module.batch_size = config["data"]["batch_size"]
        print(data_module.corpus)

    else:
        raise ConfigurationError("Filepath not recognized.")

    corpus = data_module.corpus
    print(f"N lemma scripts: {len(corpus.script_counter)}")
    print(f"N morph tags: {len(corpus.morph_tag_vocab)}")
    print(f"N morph cats: {len(corpus.morph_cat_vocab)}")

    # *==========================================================================
    # * Model
    # *==========================================================================
    print(f"\n{timer.time()} | MODEL SETUP")

    # Convert warmup steps defined as percentage of total training steps
    # to a usable integer
    if (
        config["model"].get("n_warmup_steps", False)
        and config["model"]["n_warmup_steps"] <= 1.0
        and config["model"]["n_warmup_steps"] > 0
    ):
        adj_n_warmup_steps = int(
            config["model"]["n_warmup_steps"]
            * config["trainer"]["max_epochs"]
            * len(data_module.train_dataloader())
        )

        print(f"Setting {adj_n_warmup_steps} as the total number of warmup steps.")
        print(
            f"{config['model']['n_warmup_steps']*100:.2f}% of total training steps.\n"
        )

        config["model"]["n_warmup_steps"] = adj_n_warmup_steps

    if config["architecture"].lower() == "udpipe2":
        model = UDPipe2(
            len_char_vocab=len(corpus.char_vocab),
            char_unk_idx=corpus.char_vocab[corpus.unk_token],
            char_pad_idx=corpus.char_vocab[corpus.pad_token],
            len_token_vocab=len(corpus.token_vocab),
            token_unk_idx=corpus.token_vocab[corpus.unk_token],
            token_pad_idx=corpus.token_vocab[corpus.pad_token],
            pretrained_embedding_dim=corpus.pretrained_embeddings_dim,
            n_lemma_scripts=len(corpus.script_counter),
            n_morph_tags=len(corpus.morph_tag_vocab),
            n_morph_cats=len(corpus.morph_cat_vocab),
            preprocessor_kwargs=config.preprocessor,
            **config["model"],
        )

    elif config["architecture"].lower() == "udify":
        model = UDIFY(
            len_char_vocab=len(corpus.char_vocab),
            idx_char_unk=corpus.char_vocab[corpus.unk_token],
            idx_char_pad=corpus.char_vocab[corpus.pad_token],
            idx_token_unk=corpus.token_vocab[corpus.unk_token],
            idx_token_pad=corpus.token_vocab[corpus.pad_token],
            n_lemma_scripts=len(corpus.script_counter),
            n_morph_tags=len(corpus.morph_tag_vocab),
            n_morph_cats=len(corpus.morph_cat_vocab),
            **config["model"],
        )

    elif (
        config["architecture"].lower() == "udify_finetune"
        or config["architecture"].lower() == "udifyfinetune"
    ):
        model = UDIFYFineTune(
            device=device,
            n_lemma_scripts=len(corpus.script_counter),
            n_morph_tags=len(corpus.morph_tag_vocab),
            n_morph_cats=len(corpus.morph_cat_vocab),
            **config["model"],
        )

    elif config["architecture"].lower() == "dogtag":
        model = DogTag(
            idx_char_pad=corpus.char_vocab[corpus.pad_token],
            idx_token_pad=corpus.token_vocab[corpus.pad_token],
            n_lemma_scripts=len(corpus.script_counter),
            n_morph_tags=len(corpus.morph_tag_vocab),
            n_morph_cats=len(corpus.morph_cat_vocab),
            **config["model"],
        )

    elif config["architecture"].lower() == "dogtagsmall":
        model = DogTagSmall(
            idx_char_pad=corpus.char_vocab[corpus.pad_token],
            idx_token_pad=corpus.token_vocab[corpus.pad_token],
            n_lemma_scripts=len(corpus.script_counter),
            n_morph_tags=len(corpus.morph_tag_vocab),
            n_morph_cats=len(corpus.morph_cat_vocab),
            **config["model"],
        )

    else:
        raise NotImplementedError(
            "Architecture {config['architecture'].lower()} not recognized."
        )

    # *==========================================================================
    # * Train
    # *==========================================================================
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        gpus=1 if use_cuda else 0,
        deterministic=False,
        benchmark=False,
        fast_dev_run=(
            int(config["fdev_run"])
            if config["fdev_run"] > 0 or config["fdev_run"]
            else False
        ),
        weights_summary="top",
        **config["trainer"],
    )

    trainer.logger._default_hp_metric = None

    if config.get("sanity_check", False):
        print(f"\n{timer.time()} | SANITY CHECK")

        trainer.validate(
            model,
            dataloaders=data_module.val_dataloader(
                **config["data"]["dataloader_kwargs"]
            ),
            verbose=True,
        )

    print(f"\n{timer.time()} | TRAINING")

    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(
            **config["data"]["dataloader_kwargs"]
        ),
        val_dataloaders=data_module.val_dataloader(
            **config["data"]["dataloader_kwargs"]
        ),
    )

    # *==========================================================================
    # *Test
    # *==========================================================================
    print(f"\n{timer.time()} | TESTING")
    if not (config["fdev_run"] > 0 or config["fdev_run"]):
        # If in fastdev mode, won't save a model
        # Would otherwise throw a 'PermissionError: [Errno 13] Permission denied: ...'

        print("\nTESTING")
        if config.get("save_checkpoints", True):
            # If models are being saved, load the best and apply it to the test dataset
            # Otherwise, just go straight to testing
            print(f"LOADING FROM {trainer.checkpoint_callback.best_model_path}")
            model = model.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path, strict=False
            )
            model.freeze()
            model.eval()

        test_result = trainer.test(
            model,
            datamodule=data_module.test_dataloader(
                **config["data"]["dataloader_kwargs"]
            ),
            verbose=True,
        )

        timer.end()

        return test_result

    else:
        timer.end()

        return 1


if __name__ == "__main__":

    # * WARNINGS FILTER
    #! THESE ARE KNOWN, REDUNDANT WARNINGS
    warnings.filterwarnings("ignore", message=r".*Named tensors.*")
    warnings.filterwarnings(
        "ignore", message=r".*does not have many workers which may be a bottleneck.*"
    )
    warnings.filterwarnings("ignore", message=r".*GPU available but not used .*")
    warnings.filterwarnings("ignore", message=r".*shuffle=True")
    warnings.filterwarnings("ignore", message=r".*Trying to infer .*")
    warnings.filterwarnings(
        "ignore", message=r".*DataModule.setup has already been called.*"
    )

    train()
