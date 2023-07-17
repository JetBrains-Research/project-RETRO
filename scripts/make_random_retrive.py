from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import os
import time

import torch
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config

import numpy as np

"""
"""

add_flag = ""

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
retrieve_hyperparams = config.retrieve.hyperparams

stats_path = os.path.join(paths.texts_folder, "processed-stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)

# %%

num_chunks = stats["chunks"]
knn = retrieve_hyperparams.n_knn
seq_len = config.model_hyperparameters.max_seq_len
chunks_memmap_path = os.path.join(paths.texts_folder, "train.chunks.dat")
chunk_size = 64
seq_size = seq_len // chunk_size

tt = time.time()

all_chunks_mmp = np.memmap(chunks_memmap_path, dtype=np.int32, mode="r", shape=(num_chunks, chunk_size + 1))


n_samples = 2 * seq_size * knn
num_chunks_even = n_samples * (num_chunks // n_samples)
all_chunks = np.array(all_chunks_mmp)[:num_chunks_even, :chunk_size]
all_chunks_mmp.flush()
del all_chunks_mmp
all_chunks = torch.from_numpy(all_chunks)
all_chunks = all_chunks.reshape((num_chunks_even // 2, 2 * chunk_size))
### shuffling
indices = torch.randperm(num_chunks_even // 2)
all_chunks = all_chunks[indices]

ret_db_size = (num_chunks_even // 2) // (seq_size * knn)
all_chunks = all_chunks.reshape((ret_db_size, seq_size, knn, 2 * chunk_size))

torch.save(all_chunks, os.path.join(paths.texts_folder, "train.retrieved_random_chunks.pt"))

print(f"Time used = {time.time() - tt}")
