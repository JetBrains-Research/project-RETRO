import torch

from retro_pytorch.utils import seed_all

seed_all(1111)

import argparse
import gc
import time

from retro_pytorch.dataloaders import DataLoaderFromFile, DatasetJsonl
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.train_functions import val_steps
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
args = parser.parse_args()
no_retrieve = args.no_retrieve

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(
    max_seq_len=512,  # max sequence length
    enc_dim=768,  # encoder model dimension
    enc_depth=3,  # encoder depth
    dec_dim=768,  # decoder model dimensions
    dec_depth=12,  # decoder depth
    dec_cross_attn_layers=(1, 3, 6, 9),  # decoder cross attention layers (with causal chunk cross attention)
    heads=8,  # attention heads
    dim_head=64,  # dimension per head
    dec_attn_dropout=0.25,  # decoder attention dropout
    dec_ff_dropout=0.25,  # decoder feedforward dropout
).cuda()

#%%

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

#%%

batch_size = 6
total_items = 136701

print(f"Batch size = {batch_size}")

val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_dl = DataLoaderFromFile(val_ds, batch_size=batch_size)
val_dl_iter = iter(val_dl)

fetch_neighbours = wrapper_db.fetch_neighbours
losses_val = []

# model_file = model_folder + 'retro_no_retrieve_last.pth'
model_file = model_folder + "retro_last.pth"
retro.load_state_dict(torch.load(model_file))
retro.eval()

tt = time.time()
losses_val, _ = val_steps(retro, no_retrieve, fetch_neighbours, num_val=None, val_dl_iter=val_dl_iter)

# f_val_total = open(out_folder + "losses_val_final.txt", "a")

val_avg = sum(losses_val) / len(losses_val)
print(f"Average validation loss = {val_avg}")
time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")
