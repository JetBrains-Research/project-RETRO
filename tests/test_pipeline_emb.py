from retro_pytorch.utils import seed_all

seed_all(1111)

import argparse
import json
from typing import Any

import faiss
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from retro_pytorch.dataloaders import DataLoaderFromFile, DatasetJsonl
from retro_pytorch.retrieval import SOS_ID, embed

# %%

"""
Tests distance between pre-calculated embeddings and train-pipline calculated. Calculates retrieval error rate.
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

# %%


def decode(tens: torch.Tensor) -> Any:
    mask = tens != 0
    non_zero_tensor = torch.masked_select(tens, mask)
    return tokenizer.decode(non_zero_tensor)


def print_ids(tens: torch.Tensor) -> None:
    print(decode(tens))


def faiss_read_index(path: str) -> Any:
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)


# embeddings = np.array(embeddings)

# %%

embeddings = np.memmap(
    paths.texts_folder + "train.chunks.dat.embedded", dtype=np.float32, mode="r", shape=(num_chunks, 768)
)

chunks = np.memmap(paths.texts_folder + "train.chunks.dat", dtype=np.int32, mode="r", shape=(num_chunks, 65))

# %%

batch_size = 100
val_data_path = paths.data_folder + paths.val_data_file
val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_dl = iter(DataLoaderFromFile(val_ds, batch_size=batch_size))

distances: list[float] = []
distances_pre: list[float] = []
list_of_emb: list[torch.Tensor] = []
start = 0
i = 0
for chunks_seq_batch, _ in tqdm(val_dl, total=70_000 // batch_size):
    chunks_batch = rearrange(chunks_seq_batch[:, :-1], "b (n c) -> (b n) c", c=chunk_size)
    zero_ind = ~torch.all(chunks_batch == 0, dim=-1)
    chunks_batch = chunks_batch[zero_ind]
    num_chunks_in_batch = len(chunks_batch)
    targ_emb = torch.tensor(embeddings[start : start + num_chunks_in_batch])

    sos = SOS_ID * torch.ones((chunks_batch.shape[0], 1), dtype=torch.bool)
    chunks_batch = torch.cat((sos, chunks_batch), dim=1)

    out = embed(chunks_batch).cpu()
    list_of_emb.append(out)

    dist = torch.mean(torch.norm(out - targ_emb, dim=-1)).item()
    distances.append(dist)

    start += num_chunks_in_batch
    i += 1
    # break

    if (i + 1) % 40 == 0:
        print(f"\n Mean distance = {sum(distances) / len(distances)}")
        # print(f'Mean distance gold = {sum(distances_pre)/len(distances_pre)}')
        break

emb_calculated = torch.cat(list_of_emb)
targ_emb = torch.tensor(embeddings[:start])

print(f"Mean distance = {sum(distances) / len(distances)}")
# %%
faiss_index = faiss_read_index(paths.texts_folder + "knn.index")
dist_db, indices_res_db = faiss_index.search(targ_emb, k=1)
dist, indices_res = faiss_index.search(emb_calculated, k=1)

print(f"Error rate original = {sum(dist[:, 0] > 1e-6) / len(dist)}")
print(f"Error rate = {sum(dist_db[:, 0] > 1e-6) / len(dist_db)}")

# %%
import gc

gc.collect()
torch.cuda.empty_cache()
# %%


# %%
