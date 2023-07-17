from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import os
import time
from datetime import datetime

from omegaconf import OmegaConf
from tqdm import tqdm

from retro_pytorch.dataloaders import DataLoaderFromFile, DatasetJsonl
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.train_functions import aggregate_batches, calc_loss, grad_step, val_steps, val_update
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
parser.add_argument("-config", "--config", default="config_dev.yaml", help="Config filename")
args = parser.parse_args()
no_retrieve = args.no_retrieve
config_name = args.config

# Use the arguments in your program
if no_retrieve:
    print(f"NO retrieve during training")
    add_flag = "_no_retrieve"
else:
    print(f"Retrieval would be used during training")
    add_flag = ""

"""
Training. Add flag --no-retrieve or -no if you want to train without retrieval.
It would add '_no_retrieve' to output filenames (model and train/val loss tracking)
"""

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
training_params = config.training_params
retrieve_hyperparams = config.retrieve.hyperparams
index_params = config.retrieve.hnsw_params

model_name = paths.model_name + add_flag

train_data_path = os.path.join(paths.data_folder, paths.train_data_file)
val_data_path = os.path.join(paths.data_folder, paths.val_data_file)
filename_train = os.path.join(paths.out_folder, paths.out_filename_train + add_flag + ".txt")
filename_val = os.path.join(paths.out_folder, paths.out_filename_val + add_flag + ".txt")
stats_path = os.path.join(paths.texts_folder, "processed-stats.json")
f_train = open(filename_train, "a")
f_val = open(filename_val, "a")
with open(stats_path, "r") as f:
    stats = json.load(f)

# instantiate RETRO, fit it into the TrainingWrapper with correct settings
retro = RETRO(**config.model_hyperparameters).cuda()

if no_retrieve:
    print("Freezing encoder parameters")
    for param in retro.encoder.parameters():
        param.requires_grad = False

wrapper_db = TrainingWrapper(
    retro=retro,  # path to retro instance
    knn=retrieve_hyperparams.n_knn,  # knn (2 in paper was sufficient)
    chunk_size=stats["chunk_size"],  # chunk size (64 in paper)
    documents_path=paths.data_folder,  # path to folder of text
    data_file_paths=[],
    chunks_memmap_path=os.path.join(paths.texts_folder, "train.chunks.dat"),  # path to chunks
    seqs_memmap_path=os.path.join(paths.texts_folder, "train.seq.dat"),  # path to sequence data
    doc_ids_memmap_path=os.path.join(
        paths.texts_folder, "train.doc_ids.dat"
    ),  # path to document ids per chunk (used for filtering neighbors belonging to same document)
    processed_stats_json_path=stats_path,
    knn_extra_neighbors=retrieve_hyperparams.knn_extra_neighbors,  # num extra neighbors to fetch
    precalculate_knn=False,
    index_params=index_params,
)

# %%

### setting up number of steps.
freq_val = training_params.freq_val  # frequency of validation
num_val = training_params.num_val  # number of validation steps
batch_size = training_params.batch_size
batch_size_val = training_params.batch_size_val
batch_accumulation = training_params.batch_accumulation
total_items = 1367016

accumulate_steps = (
    batch_accumulation // batch_size if batch_accumulation % batch_size == 0 else batch_accumulation // batch_size + 1
)
batch_accumulation = accumulate_steps * batch_size
total_steps = total_items // batch_accumulation
warmup_steps = total_steps // 25  ### 4% for warmup
lr = training_params.lr  # learning rate

### Ensure that validation is performed after taking the gradient step.
freq_val = (freq_val // accumulate_steps) * accumulate_steps

# loading data and optimization functions.
train_ds = DatasetJsonl(train_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
train_dl = DataLoaderFromFile(train_ds, batch_size=batch_size)
val_dl = DataLoaderFromFile(val_ds, batch_size=batch_size_val)

optim, scheduler = wrapper_db.get_optimizer(warmup_steps=warmup_steps, training_steps=total_steps, lr=lr, wd=0.01)
scheduler.step()
fetch_neighbours = wrapper_db.fetch_neighbours
fetch_random_chunk = wrapper_db.fetch_random_chunk
generate_pure_random_chunk = wrapper_db.generate_pure_random_chunk

# %%

losses_train: list[float] = []
losses_val: list[list[float]] = []
losses_val_pure_rnd: list[float] = []
losses_val_chunk_rnd: list[float] = []
train_steps = 0
max_val_loss = 10000.0

text_start = f"\n------- NEW TRAINING {str(datetime.now())}, batch size = {batch_size}, batch_accum = {batch_accumulation}, warmup steps = {warmup_steps}, validation frequency = {freq_val}, learining rate = {lr}-------\n"
f_train.write(text_start)
f_val.write(text_start)
print(text_start)

tt = time.time()

saved_ind = 0
saved_last_ind = 0
val_dl_iter = iter(val_dl)

for train_steps, (seq, docs) in enumerate(tqdm(train_dl, total=total_items // batch_size), start=1):

    loss = calc_loss(seq, docs, retro, no_retrieve, fetch_neighbours)

    if train_steps % accumulate_steps == 0:
        grad_step(optim, scheduler, loss, losses_train, f_train)

    if train_steps % freq_val == 0:

        print("------ Validation ------")
        f_train.flush()
        retro.eval()

        aggregate, val_step = aggregate_batches(val_dl_iter, num_val)
        # losses_val_cur, val_step = val_steps(retro, no_retrieve, fetch_neighbours, num_val, val_dl_iter)
        losses_val_cur = val_steps(retro, no_retrieve, fetch_neighbours, aggregate)
        if not no_retrieve:
            losses_val_rnd_cur = val_steps(retro, no_retrieve, fetch_random_chunk, aggregate)
            losses_val_pure_rnd_cur = val_steps(retro, no_retrieve, generate_pure_random_chunk, aggregate)
        else:
            losses_val_rnd_cur = losses_val_cur
            losses_val_pure_rnd_cur = losses_val_cur

        max_val_loss, saved_ind, saved_last_ind, val_dl_iter = val_update(
            retro,
            losses_val,
            [losses_val_cur, losses_val_rnd_cur, losses_val_pure_rnd_cur],
            paths.model_folder,
            model_name,
            val_dl_iter,
            f_val,
            max_val_loss,
            saved_ind,
            saved_last_ind,
        )

        if val_step < num_val:
            print("----- Reloading val dataset ------")
            val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
            val_dl = DataLoaderFromFile(val_ds, batch_size=batch_size_val)
            val_dl_iter = iter(val_dl)

        retro.train()

time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")

# %%
