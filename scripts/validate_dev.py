import torch

from retro_pytorch.utils import seed_all

seed_all(1111)

import argparse
import gc
import json
import os
import time

import numpy as np
from omegaconf import OmegaConf

from retro_pytorch.dataloaders import DataLoaderFromFile, DatasetJsonl
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.train_functions import aggregate_batches, calc_loss, val_steps
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
parser.add_argument("-config", "--config", default="config_dev.yaml", help="Config filename")
args = parser.parse_args()
no_retrieve = args.no_retrieve
config_name = args.config

# # loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
retrieve_hyperparams = config.retrieve.hyperparams
training_params = config.training_params
index_params = config.retrieve.hnsw_params
stats_path = os.path.join(paths.texts_folder, "processed-stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(**config.model_hyperparameters).cuda()
if no_retrieve:
    print("Freezing encoder parameters")
    for param in retro.encoder.parameters():
        param.requires_grad = False
#%%

val_data_path = os.path.join(paths.data_folder, paths.val_data_file)
train_data_path = os.path.join(paths.data_folder, paths.train_data_file)

gc.collect()
torch.cuda.empty_cache()

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


batch_size_val = 1
total_items = 136701

print(f"Batch size = {batch_size_val}")

val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_dl = DataLoaderFromFile(val_ds, batch_size=batch_size_val)
val_dl_iter = iter(val_dl)

train_ds = DatasetJsonl(train_data_path, cnunk_size=64, seq_length=512, pad_id=0)
train_dl = DataLoaderFromFile(train_ds, batch_size=batch_size_val)
train_dl_iter = iter(train_dl)

fetch_neighbours = wrapper_db.fetch_neighbours
fetch_random_chunk = wrapper_db.fetch_random_chunk
generate_pure_random_chunk = wrapper_db.generate_pure_random_chunk

losses_val: list[list[float]] = []
losses_val_pure_rnd: list[float] = []
losses_val_chunk_rnd: list[float] = []

# model_file = model_folder + 'retro_no_retrieve_last.pth'
model_file = paths.model_folder + "retro_last_2.pth"
retro.load_state_dict(torch.load(model_file))
retro.eval()

tt = time.time()  ## TODO rewrite. Now it follows training pipeline, but needs to load whole dataset into memory
print("Loading dataset into memory")
val_avgs_all = []
num_val = 10

for val_step, (seq, docs) in enumerate(train_dl_iter, start=1):

    val_losses = []
    for fetch_fun, fetch_name in zip(
        [fetch_neighbours, fetch_random_chunk, generate_pure_random_chunk],
        ["Fetch neigbours", "Fetch random", "Generate pure random"],
    ):

        # print(f'Validating on {fetch_name}')
        loss = calc_loss(seq, docs, retro, False, fetch_fun)
        val_losses.append(loss.item())

    print(f"Loss = {val_losses}")
    if val_step >= num_val:
        break

# for i in range(10):
#
#     print(f' ----------- {i} ------------')
#     aggregate, val_step = aggregate_batches(val_dl_iter, 1)
#     val_avgs = []
#     for fetch_fun, fetch_name in zip([fetch_neighbours, fetch_random_chunk, generate_pure_random_chunk], ['Fetch neigbours', 'Fetch random', 'Generate pure random']):
#
#         print(f'Validating on {fetch_name}')
#         losses_val = val_steps(retro, no_retrieve, fetch_fun, aggregate, verbose=False)
#         val_avg = sum(losses_val) / len(losses_val)
#         val_avgs.append(val_avg)
#         #print(f"Average validation loss on = {val_avg}")
#
#     print(val_avgs)
#     val_avgs_all.append(val_avgs)
#
# val_avgs_all = np.array(val_avgs_all)
#
# val_avgs_all = np.mean(val_avgs_all, axis=0)
#
# print(f'Final = {val_avgs_all}')


time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")
