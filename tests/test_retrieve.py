import argparse
from typing import Any

import pandas as pd
import torch
from einops import rearrange
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import os

from retro_pytorch.dataloaders import DataLoaderFromFile, DatasetJsonl
from retro_pytorch.retrieval import embed
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.training import TrainingWrapper

# %%

parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config

print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths

train_data_path = os.path.join(paths.data_folder, paths.train_data_file)
val_data_path = os.path.join(paths.data_folder, paths.val_data_file)

tokenizer = AutoTokenizer.from_pretrained(paths.encoder_path)

"""
Creates embeddings and finding knns for each chuncks in dataset
"""

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(**config.model_hyperparameters).cuda()

# %%

wrapper_db = TrainingWrapper(
    retro=retro,  # path to retro instance
    knn=2,  # knn (2 in paper was sufficient)
    chunk_size=64,  # chunk size (64 in paper)
    documents_path=paths.data_folder,  # path to folder of text
    data_file_paths=[],
    chunks_memmap_path=os.path.join(paths.texts_folder, "train.chunks.dat"),  # path to chunks
    seqs_memmap_path=os.path.join(paths.texts_folder, "train.seq.dat"),  # path to sequence data
    doc_ids_memmap_path=paths.texts_folder
    + "train.doc_ids.dat",  # path to document ids per chunk (used for filtering neighbors belonging to same document)
    processed_stats_json_path=os.path.join(paths.texts_folder, "processed-stats.json"),
    # max_chunks = n_chuncks,                        # maximum cap to chunks
    # max_seqs = n_chuncks//5,                            # maximum seqs
    knn_extra_neighbors=100,  # num extra neighbors to fetch
    max_index_memory_usage="10G",
    current_memory_available="32G",
)

# %%


def decode(tens: torch.Tensor) -> Any:
    return tokenizer.batch_decode(tens, skip_special_tokens=True)


def print_ids(tens: torch.Tensor) -> None:
    print(decode(tens))


def seq_distance(seq1: torch.Tensor, seq2: torch.Tensor) -> torch.Tensor:
    emb1 = embed(seq1, return_cls_repr=False)
    emb2 = embed(seq2, return_cls_repr=False)

    dist = torch.norm(emb1 - emb2, dim=-1)  # torch.mean().item()

    return dist


val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_dl = iter(DataLoaderFromFile(val_ds, batch_size=1))

# obtaining some reference chunk
for i in range(7):
    seq, docs = next(val_dl)

seq = seq[0].cuda()
chunks_tmp = torch.chunk(seq[:-1], chunks=8, dim=0)
reference_chunk = chunks_tmp[6]
random_chunk = torch.randint_like(reference_chunk, 0, 30000)
print("----Reference chunk------")
print("".join(decode(reference_chunk.unsqueeze(0))))
print("----Random sequence chunk------")
print("".join(decode(random_chunk.unsqueeze(0))))


val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_dl = iter(DataLoaderFromFile(val_ds, batch_size=100))
# orig_dl = iter(wrapper_db.get_dataloader(batch_size=100, shuffle=False))

fetch_neighbours = wrapper_db.fetch_neighbours

# seq_orig, retrieved_orig_all = next(orig_dl)
# if torch.all(seq_orig == seq_all).item():
#     print('sequences are equal')
# else:
#     print('ERROR!!!!')

"""
Calculates the distances between seq and retrieve, writes csv to demostrate retrieves
"""
chunk_size = 64
seq, docs = next(val_dl)
retrieved = fetch_neighbours(seq, doc_except=docs)
chunks = rearrange(seq[:, :-1], "b (n c) -> (b n) c", c=chunk_size)
mask = ~torch.all(chunks == 0, dim=-1)
chunks = chunks[mask]
retrieved = rearrange(retrieved[:, :, 0, :64], "b n c -> (b n) c")
retrieved = retrieved[mask]
reference_chunk = reference_chunk.repeat(chunks.size(0), 1)
random_chunk = random_chunk.repeat(chunks.size(0), 1)

distances = seq_distance(chunks, retrieved)
distances_control = seq_distance(chunks, reference_chunk)
distances_random = seq_distance(chunks, random_chunk)
dist_ratio = distances / (distances_control + 1e-3)
dist_ratio_random = distances / (distances_random + 1e-3)

chunks_decoded = decode(chunks)
retrieved_decoded = decode(retrieved)

data_dict = {
    "Query": chunks_decoded,
    "Retrieved": retrieved_decoded,
    "Distance": distances.cpu(),
    "Dist to reference": distances_control.cpu(),
    "Dist to random": distances_random.cpu(),
    "Dist ratio": dist_ratio.cpu(),
    "Dist ratio to random": dist_ratio_random.cpu(),
}
retrieve_examples = pd.DataFrame(data_dict)

retrieve_examples.to_csv(os.path.join(paths.out_folder, "retrieved_examples.csv"))

distances[distances != 0].mean()
