from retro_pytorch.utils import seed_all

seed_all(1111)

import argparse
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from retro_pytorch.retrieval import SOS_ID, embed

# %%

"""
Tests the match between pre-calculated embeddings and procedure used in train 
"""


parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config

print(f"Loading configs from {config_name} file")
conf_load = OmegaConf.load(config_name)
paths = conf_load.paths

stats = json.load(open(os.path.join(paths.texts_folder, "processed-stats.json")))
num_chunks = stats["chunks"]

#%%

embeddings = np.memmap(
    os.path.join(paths.texts_folder, "train.chunks.dat.embedded"), dtype=np.float32, mode="r", shape=(num_chunks, 768)
)

chunks = np.memmap(os.path.join(paths.texts_folder, "train.chunks.dat"), dtype=np.int32, mode="r", shape=(num_chunks, 65))

#%%

all_ind = np.arange(num_chunks)
n_tot = 32000
batch_size = 4000
n_tot = (n_tot // batch_size) * batch_size
rand_ind = np.random.choice(all_ind, n_tot, replace=False)
rand_ind = np.array_split(rand_ind, n_tot // batch_size)  #!!!!!! This is not random!

#%%
distances = []
for i, indices in tqdm(enumerate(rand_ind), total=len(rand_ind)):
    chunk_in = torch.tensor(chunks[indices])
    targ_emb = torch.tensor(embeddings[indices])

    sos = SOS_ID * torch.ones((chunk_in.shape[0], 1), dtype=torch.bool)
    chunk_in = torch.cat((sos, chunk_in), dim=1)

    out = embed(chunk_in[:, :-1]).cpu()

    dist = torch.mean(torch.norm(out - targ_emb, dim=-1)).item()
    distances.append(dist)
#%%

print(f"Mean distance = {sum(distances)/len(distances)}")

#%%
