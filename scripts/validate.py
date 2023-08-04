from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import os
import time
from datetime import datetime
import numpy as np
import torch

from omegaconf import OmegaConf
from tqdm import tqdm

from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
no_retrieve = args.no_retrieve
config_name = args.config

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
training_params = config.training_params
retrieve_hyperparams = config.retrieve.hyperparams
index_params = config.retrieve.hnsw_params

# Use the arguments in your program
if no_retrieve:
    print("NO retrieve during training")
    add_flag = "_no_retrieve"
else:
    print("Retrieval would be used during training")
    add_flag = ""

add_flag = add_flag + "_star"
add_flag += "_conc_proj"
knn_path_train = os.path.join(paths.texts_folder, "knn_from_project.dat")
if not no_retrieve:
    config.model_hyperparameters.max_seq_len = 2*config.model_hyperparameters.max_seq_len
    on_project = True
    print("Training on the retrieval from the projects")

#add_flag += "_dev"
"""
Training. Add flag --no-retrieve or -no if you want to train without retrieval.
It would add '_no_retrieve' to output filenames (model and train/val loss tracking)
"""

#%%

model_name = paths.model_name + add_flag

train_data_path = os.path.join(paths.data_folder, paths.train_data_file)
val_data_path = os.path.join(paths.data_folder, paths.val_data_file)
filename_val = os.path.join(paths.out_folder, paths.out_filename_val + add_flag + "_final.txt")
stats_path = os.path.join(paths.texts_folder, paths.processed_stats_filename)

with open(stats_path, "r") as f:
    stats = json.load(f)

# config.model_hyperparameters.dec_cross_attn_layers = eval(config.model_hyperparameters.dec_cross_attn_layers)

# import torch
retro = RETRO(**config.model_hyperparameters).cuda()
# model_file = paths.model_folder + "retro_concat_last_1epoch.pth"
# retro.load_state_dict(torch.load(model_file))
retro.eval()

#%%

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
    knn_memmap_path_option=None,# knn_path_optional,  ## used for additional validaton purposes
    split_meta_path=os.path.join(paths.texts_folder, "split_meta_dict.json"),
    knn_extra_neighbors=retrieve_hyperparams.knn_extra_neighbors,  # num extra neighbors to fetch
    precalculate_knn=False,
    index_params=index_params,
)

fetch_random_chunk = wrapper_db.fetch_random_chunk
generate_pure_random_chunk = wrapper_db.generate_pure_random_chunk

batch_size_val = training_params.batch_size_val
val_dl = iter(wrapper_db.get_dataloader(split="val", batch_size=batch_size_val, shuffle=True))
# %%

losses_val: list[list[float]] = []

text_start = f"\n------- FINAL VALIDATION {str(datetime.now())}, batch size = {batch_size_val} -------\n"
# f_val = open(filename_val, "a")
# f_val.write(text_start)
print(text_start)

losses_val = []

tt = time.time()

with torch.no_grad():
    for step, (seq, ret) in enumerate(tqdm(val_dl, ncols=80), start=1):
        seq = seq.cuda()
        if no_retrieve:
            ### TODO add control - training without retrieve, validating with it.
            loss = retro(seq, retrieved=None, return_loss=True)
            losses = [loss.item()]
        else:
            # loss = retro(seq, retrieved=ret.cuda(), return_loss=True)
            loss_none = retro(seq, retrieved=None, return_loss=True) ### TODO Check, what is with positional encoding here?
            losses = [loss.item(), loss_none.item()]

            for fetch_neighbours in [fetch_random_chunk, generate_pure_random_chunk]:
                retrieved = fetch_neighbours(seq)
                loss = retro(seq, retrieved=retrieved.cuda(), return_loss=True)
                losses.append(loss.item())

        losses_val.append(losses)

        step += 1
        if step >= 10:
            break

losses_val = np.array(losses_val)
filename_val = os.path.join(paths.out_folder, paths.out_filename_val + add_flag + "_final.npy")
np.save(filename_val, losses_val)

time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")

print(losses_val.shape)

#%%

