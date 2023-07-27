from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import os
import time
from datetime import datetime

import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from retro_pytorch.retrieval import SOS_ID
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.train_functions import aggregate_batches, grad_step, val_steps, val_update
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config


from retro_pytorch.retrieval import embed


def chunk_distance(seq1, seq2):
    emb1 = embed(seq1, return_cls_repr=True)
    emb2 = embed(seq2, return_cls_repr=True)

    dist = torch.norm(emb1 - emb2, dim=-1)

    return dist


## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
training_params = config.training_params
retrieve_hyperparams = config.retrieve.hyperparams
index_params = config.retrieve.hnsw_params


train_data_path = os.path.join(paths.data_folder, paths.train_data_file)
val_data_path = os.path.join(paths.data_folder, paths.val_data_file)
stats_path = os.path.join(paths.texts_folder, "processed-stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)

# config.model_hyperparameters.dec_cross_attn_layers = eval(config.model_hyperparameters.dec_cross_attn_layers)

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(**config.model_hyperparameters).cuda()

with open(paths.data_folder + "split_doc_dict.json", "r") as file:
    split_doc_dict = json.load(file)
with open(paths.data_folder + "split_ind_dict.json", "r") as file:
    split_ind_dict = json.load(file)

split_len_dic = {key: len(value) for key, value in split_ind_dict.items()}
#%%

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
    knn_memmap_path=os.path.join(paths.texts_folder, "knn_per_project.dat"),
    knn_memmap_path_option=os.path.join(paths.texts_folder, "knn_from_all.dat"),
    split_meta_path=os.path.join(paths.texts_folder, "split_meta_dict.json"),
    knn_extra_neighbors=retrieve_hyperparams.knn_extra_neighbors,  # num extra neighbors to fetch
    precalculate_knn=False,
    index_params=index_params,
)

fetch_random_chunk = wrapper_db.fetch_random_chunk
generate_pure_random_chunk = wrapper_db.generate_pure_random_chunk

num_chunks = stats["chunks"]
chunk_size = stats["chunk_size"]

# %%
split = "train"
batch_size = 300
dl = iter(wrapper_db.get_dataloader(split=split, batch_size=batch_size, shuffle=True))
# dl = iter(wrapper_db.get_dataloader(split='val', batch_size=batch_size, shuffle = True))
# %%

"""
Calculates the distances between seq and retrieve, writes csv to demostrate retrieves
"""

dist_arr = []
num_steps = 60

for train_steps, (seq, ret1, ret2) in enumerate(tqdm(dl, total=num_steps), start=1):

    chunks_batch = rearrange(seq[:, :-1], "b (n c) -> (b n) c", c=chunk_size)
    zero_ind = ~torch.all(chunks_batch == 0, dim=-1)
    chunks_batch = chunks_batch[zero_ind].cuda()

    retr_list = [ret1.cuda(), ret2.cuda()]

    for fetch_neighbours in [fetch_random_chunk, generate_pure_random_chunk]:
        retr_list.append(fetch_neighbours(seq))

    retr_list = torch.stack(retr_list)
    retr_list = retr_list[:, :, :, 0, :64]
    retr_list = retr_list.view(retr_list.size(0), retr_list.size(1) * retr_list.size(2), retr_list.size(3))
    retr_list = retr_list[:, zero_ind]

    sos = SOS_ID * torch.ones((chunks_batch.size(0), 1), dtype=torch.bool).cuda()
    chunks_batch = torch.cat((sos, chunks_batch), dim=1)

    sos = SOS_ID * torch.ones((4, chunks_batch.size(0), 1), dtype=torch.bool).cuda()
    retr_list = torch.cat((sos, retr_list), dim=-1)

    dist = torch.stack([chunk_distance(chunks_batch, retrieved).cpu() for retrieved in retr_list])
    dist = dist.permute(1, 0)
    dist_arr.append(dist)

    if train_steps > num_steps:
        break

dist_arr = torch.cat(dist_arr, dim=0)
dist_mean = torch.mean(dist_arr, dim=0)
dist_std = torch.std(dist_arr, dim=0)
print(f"\n Counted {dist_arr.size(0)} chunks in shuffeled {split} split")
print(f"Mean distances {dist_mean.tolist()}")
print(f"Std distances {dist_std.tolist()}")
#%%

main_dist = dist_arr[:, 0].unsqueeze(1)
dist_diffs = dist_arr[:, 1:] - main_dist
dist_diffs_mean = torch.mean(dist_diffs, dim=0)
dist_diffs_std = torch.std(dist_diffs, dim=0)
print(f"Mean difference between distances {dist_diffs_mean.tolist()}")
print(f"Std difference between distances {dist_diffs_std.tolist()}")

#%%
