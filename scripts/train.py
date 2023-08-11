from retro_pytorch.utils import seed_all

seed_all(1111)
# import torch
# torch.set_float32_matmul_precision('medium')

import argparse
import json
import os
import time
from datetime import datetime

from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from retro_pytorch.lightning_setup import LitModel
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
parser.add_argument("-project", "--project", default="dev", help="Project name")
args = parser.parse_args()
no_retrieve = args.no_retrieve
config_name = args.config
project_name = args.project

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
training_params = config.training_params
retrieve_hyperparams = config.retrieve.hyperparams
index_params = config.retrieve.hnsw_params

add_flag = "_conc_proj"
n_prepend = (
    config.model_hyperparameters.n_prepend
)  # 1 - take only retrieveв chunk, 2 - retrieved chunk and its continuation
config.model_hyperparameters.max_seq_len = (n_prepend + 1) * config.model_hyperparameters.max_seq_len

if no_retrieve:
    add_flag += "_no-ret"
    print("No retrieval during training")
else:
    print("Retrieval would be used during training")

knn_path_train = os.path.join(paths.texts_folder, "knn_from_project.dat")

"""
Training. Add flag --no-retrieve or -no if you want to train without retrieval.
It would add '_no-ret' to output filenames (model and train/val loss tracking)
"""

#%%

model_name = paths.model_name + add_flag
train_data_path = os.path.join(paths.data_folder, paths.train_data_file)
val_data_path = os.path.join(paths.data_folder, paths.val_data_file)
stats_path = os.path.join(paths.texts_folder, paths.processed_stats_filename)
with open(stats_path, "r") as f:
    stats = json.load(f)

# config.model_hyperparameters.dec_cross_attn_layers = eval(config.model_hyperparameters.dec_cross_attn_layers)
retro = RETRO(**config.model_hyperparameters).cuda()
# model_file = paths.model_folder + "retro_concat_last_1epoch.pth"
# retro.load_state_dict(torch.load(model_file))
retro.train()

wrapper_db = TrainingWrapper(
    retro=retro,  # path to retro instance
    knn=retrieve_hyperparams.n_knn,  # knn (2 in paper was sufficient)
    chunk_size=config.retrieve.chunk_size,  # chunk size (64 in paper)
    documents_path=paths.data_folder,  # path to folder of text
    data_file_paths=[],
    chunks_memmap_path=os.path.join(paths.texts_folder, "train.chunks.dat"),  # path to chunks
    seqs_memmap_path=os.path.join(paths.texts_folder, "train.seq.dat"),  # path to sequence data
    doc_ids_memmap_path=os.path.join(
        paths.texts_folder, "train.doc_ids.dat"
    ),  # path to document ids per chunk (used for filtering neighbors belonging to same document)
    processed_stats_json_path=stats_path,
    knn_memmap_path=knn_path_train,  # used for the training
    knn_memmap_path_option=None,  # knn_path_optional,  ## used for additional validaton purposes
    split_meta_path=os.path.join(paths.texts_folder, "split_meta_dict.json"),
    knn_extra_neighbors=retrieve_hyperparams.knn_extra_neighbors,  # num extra neighbors to fetch
    precalculate_knn=False,
    index_params=index_params,
)

## Getting different fetching functions
fetch_random_chunk = wrapper_db.fetch_random_chunk
generate_pure_random_chunk = wrapper_db.generate_pure_random_chunk
fetch_self_ret = wrapper_db.fetch_self_ret  # just returning retrieve
fetch_previous = wrapper_db.fetch_previous

### setting up number of steps.
freq_val = training_params.freq_val  # frequency of validation
num_val = training_params.num_val  # number of validation steps
batch_size = training_params.batch_size + n_prepend
batch_size_val = training_params.batch_size_val
batch_accumulation = training_params.batch_accumulation
accumulate_batches = (
    batch_accumulation // batch_size if batch_accumulation % batch_size == 0 else batch_accumulation // batch_size + 1
)
### Ensure that validation is performed after taking the gradient step.
freq_val = (freq_val // accumulate_batches) * accumulate_batches

shuffle_train = not no_retrieve
print(
    f"Train shuffle = {shuffle_train}"
)  ## TODO rewrite dataloader, so, it can load concequent sequences. For now it is just a shortcut - not to shuffle train DS.
val_dl = wrapper_db.get_dataloader(split="val", batch_size=batch_size_val, shuffle=False)
train_dl = wrapper_db.get_dataloader(split="train", batch_size=batch_size, shuffle=shuffle_train)

total_batches = len(train_dl)
training_params["training_steps"] = total_batches // accumulate_batches
training_params["warmup_steps"] = training_params["training_steps"] // 25  ### 4% for warmup

text_start = (
    f"\n------- NEW TRAINING {str(datetime.now())}, batch size = {batch_size},"
    f" batch_accum = {batch_accumulation}, warmup steps = {training_params['warmup_steps']}, validation frequency = {freq_val}, learining rate = {training_params['lr']}-------\n"
)
print(text_start)

wandb_logger = WandbLogger(project=project_name)  # , name='', resume="6af28sze"
wandb_logger.experiment.config["batch_size"] = batch_size

checkpoint_callback = ModelCheckpoint(
    monitor="val/loss",  # Metric to monitor for checkpointing
    dirpath=paths.model_folder,  # Directory where checkpoints will be saved
    filename=f"checkpoint{add_flag}" + "-{val/loss:.2f}",  # Checkpoint filename format
    save_top_k=2,  # Save the best checkpoint based on the monitor metric
    save_last=True,  # Save the last model
    mode="min",  # Mode of the monitor metric ('min' or 'max')
)
checkpoint_callback.CHECKPOINT_NAME_LAST = "last" + add_flag

model = LitModel(
    retro,
    train_parameters=training_params,
    no_retrieve=no_retrieve,
    n_prepend=n_prepend,  # 1 - prepend only retrieve, without continuation
    self_retr_fun=fetch_self_ret,
    retrieve_functions=[fetch_self_ret, fetch_previous],
    fun_names=["", "_previos_seq"],
)

trainer = Trainer(
    accelerator="gpu",
    max_epochs=training_params.num_epochs,
    # fast_dev_run = 100,
    val_check_interval=freq_val,
    limit_val_batches=num_val,
    accumulate_grad_batches=accumulate_batches,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    default_root_dir=paths.model_folder,
    log_every_n_steps=1,
)

tt = time.time()

trainer.fit(model, train_dl, val_dl)  # , ckpt_path=paths.model_folder+"last_conc_proj_no-ret.ckpt"
wandb_logger.finalize()

time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")
