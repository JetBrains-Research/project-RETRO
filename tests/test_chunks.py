from retro_pytorch.utils import seed_all

seed_all(1111)

import argparse
import json

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from retro_pytorch.dataloaders import DataLoaderFromFile, DatasetJsonl

# %%

"""
Tests that indexed chunks are the same as we get in training procedure 
"""

parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config_dev.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config

print(f"Loading configs from {config_name} file")
conf_load = OmegaConf.load(config_name)
paths = conf_load.paths

tokenizer = AutoTokenizer.from_pretrained(paths.encoder_path)
stats = json.load(open(paths.texts_folder + "processed-stats.json"))
num_chunks = stats["chunks"]
chunk_size = 64

#%%

chunks = np.memmap(paths.texts_folder + "train.chunks.dat", dtype=np.int32, mode="r", shape=(num_chunks, 65))

batch_size = 10000
val_data_path = paths.data_folder + paths.val_data_file
val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_dl = iter(DataLoaderFromFile(val_ds, batch_size=batch_size))

#%%

is_equal = []
start = 0
for chunks_seq_batch, _ in tqdm(val_dl, total=70_000 // batch_size):
    chunks_batch = rearrange(chunks_seq_batch[:, :-1], "b (n c) -> (b n) c", c=chunk_size)
    zero_ind = ~torch.all(chunks_batch == 0, dim=-1)
    chunks_batch = chunks_batch[zero_ind]
    num_chunks_in_batch = len(chunks_batch)

    # for chunks_seq in chunks_batch:
    is_equal_batch = torch.all(chunks_batch == torch.tensor(chunks[start : start + num_chunks_in_batch, :64])).item()
    is_equal.append(is_equal_batch)
    # print(is_equal_batch)
    start += num_chunks_in_batch
    #    if not all(is_equal):
    #        break
    if not all(is_equal):
        break

#%%

print(all(is_equal))
print(len(is_equal))

#%%
