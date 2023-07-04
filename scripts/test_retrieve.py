import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from retro_pytorch.data import DataLoader_from_file, DatasetJsonl
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.training import TrainingWrapper

# %%

model_path = "../../models/CodeT5p-220M/"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def print_ids(tens):
    mask = tens != 0
    non_zero_tensor = torch.masked_select(tens, mask)
    print(tokenizer.decode(non_zero_tensor))


# %%


"""
Creates embeddings and finding knns for each chuncks in dataset
"""

import gc

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(
    max_seq_len=512,  # max sequence length
    enc_dim=768,  # encoder model dimension 896
    enc_depth=3,  # encoder depth
    dec_dim=768,  # decoder model dimensions
    dec_depth=12,  # decoder depth
    dec_cross_attn_layers=(
        1,
        3,
        6,
        9,
    ),  # decoder cross attention layers (with causal chunk cross attention)
    heads=8,  # attention heads
    dim_head=64,  # dimension per head
    dec_attn_dropout=0.25,  # decoder attention dropout
    dec_ff_dropout=0.25,  # decoder feedforward dropout
).cuda()

# %%

texts_folder = "../../data/texts_folder/"
data_folder = "../../data/full_dataset/"
model_folder = "../../data/models/"
out_folder = "../out_dir/"

tain_data_path = data_folder + "train.jsonl"
val_data_path = data_folder + "val.jsonl"

gc.collect()
torch.cuda.empty_cache()

wrapper_db = TrainingWrapper(
    retro=retro,  # path to retro instance
    knn=2,  # knn (2 in paper was sufficient)
    chunk_size=64,  # chunk size (64 in paper)
    documents_path=data_folder,  # path to folder of text
    data_file_paths=[],
    chunks_memmap_path=texts_folder + "train.chunks.dat",  # path to chunks
    seqs_memmap_path=texts_folder + "train.seq.dat",  # path to sequence data
    doc_ids_memmap_path=texts_folder
    + "train.doc_ids.dat",  # path to document ids per chunk (used for filtering neighbors belonging to same document)
    processed_stats_json_path=texts_folder + "processed-stats.json",
    # max_chunks = n_chuncks,                        # maximum cap to chunks
    # max_seqs = n_chuncks//5,                            # maximum seqs
    knn_extra_neighbors=100,  # num extra neighbors to fetch
    max_index_memory_usage="10G",
    current_memory_available="32G",
)


# %%
from retro_pytorch.retrieval import bert_embed


def seq_distance(seq1, seq2):
    emb1 = bert_embed(seq1.unsqueeze(0), return_cls_repr=False, isdecoder=True)

    emb2 = bert_embed(seq2.unsqueeze(0), return_cls_repr=False, isdecoder=True)

    return torch.cdist(emb1, emb2)[0]


# %%

batch_size = 1

train_ds = DatasetJsonl(tain_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)

train_dl = iter(DataLoader_from_file(train_ds, batch_size=batch_size))
val_dl = iter(DataLoader_from_file(val_ds, batch_size=batch_size))
orig_dl = iter(wrapper_db.get_dataloader(batch_size=1, shuffle=False))

optim = wrapper_db.get_optimizer(lr=3e-4, wd=0.01)
fetch_neighbours = wrapper_db.fetch_neighbours

# %%

print("----New-----")

seq, docs = next(val_dl)
retrieved = fetch_neighbours(seq, doc_except=docs)
seq, retrieved = seq[0], retrieved[0]
chunks = torch.chunk(seq[:-1], chunks=8, dim=0)
print(torch.cat([seq_distance(chunks[i], retrieved[i, 0, :64]) for i in range(8)]))

print("----Original-----")
seq_orig, retrieved_orig = next(orig_dl)
seq_orig, retrieved_orig = seq_orig[0], retrieved_orig[0]
chunks_orig = torch.chunk(seq_orig[0:-1], chunks=8, dim=0)
print(torch.cat([seq_distance(chunks_orig[i], retrieved_orig[i, 0, :64]) for i in range(8)]))

# %%

diffs = torch.tensor([]).cpu()

for i in range(400):
    seq, docs = next(val_dl)
    retrieved = fetch_neighbours(seq, doc_except=docs)
    seq, retrieved = seq[0], retrieved[0]
    chunks = torch.chunk(seq[:-1], chunks=8, dim=0)
    dist = torch.cat([seq_distance(chunks[i], retrieved[i, 0, :64]) for i in range(8)])

    seq_orig, retrieved_orig = next(orig_dl)
    seq_orig, retrieved_orig = seq_orig[0], retrieved_orig[0]
    chunks_orig = torch.chunk(seq_orig[0:-1], chunks=8, dim=0)
    dist_orig = torch.cat([seq_distance(chunks_orig[i], retrieved_orig[i, 0, :64]) for i in range(8)])

    diff = (dist - dist_orig)[dist > 1e-4]
    diffs = torch.cat((diffs, diff.cpu()))

    if (i + 1) % 10 == 0:
        print(i)
        print(torch.mean(diffs))

# %%
print("Total differences")
print(torch.mean(diffs))
print(torch.std(diffs))

# %%

chunck_num = 0

print("!!!--------------New version--------------!!!!")
print("--------------Chunk--------------")
print_ids(chunks[chunck_num])
print("--------------Retrieved--------------")
print_ids(retrieved[chunck_num, 0, :64])
print("--------------Second--------------")
print_ids(retrieved[chunck_num, 1, :64])

print("!!!--------------Original version--------------!!!!")
print("--------------Chunk--------------")
print_ids(chunks_orig[chunck_num])
print("--------------Retrieved--------------")
print_ids(retrieved_orig[chunck_num, 0, :64])
print("--------------Second--------------")
print_ids(retrieved_orig[chunck_num, 1, :64])

# %%
chunck_num = 0

print("--------------Next--------------")
print_ids(chunks[chunck_num + 1])
print("--------------Retrieved continue 1--------------")
print_ids(retrieved[chunck_num, 0, 64:])
print("--------------Retrieved continue 2--------------")
print_ids(retrieved[chunck_num, 1, 64:])

# print('--------------Chunk--------------')
# print_ids(chunks[chunck_num])
# print('--------------Retrieved--------------')
# print_ids(retrieved[chunck_num, 0, :64])
