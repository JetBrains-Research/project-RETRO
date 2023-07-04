import random

import numpy as np
import torch

seed = 1111
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

import time
from datetime import datetime

from tqdm import tqdm

from retro_pytorch.data import DataLoader_from_file, Dataset_jsonl
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.training import TrainingWrapper

# %%

### Important notes!!!!
### !!!! I had to change autofaiss files C:\Users\Timur.Galimzyanov\anaconda3\lib\site-packages\autofaiss\indices\index_utils.py
### line 23     with NamedTemporaryFile() as tmp_file: -->> with NamedTemporaryFile(delete=False) as tmp_file:

# %%


"""
Creates embeddings and finding knns for each chuncks in dataset
"""

import gc

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(
    max_seq_len=512,  # max sequence length
    enc_dim=768,  # encoder model dimension
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
model_name = "retro"

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

batch_size = 14
batch_accumulation = 64
accumulate_steps = accumulate_steps = (
    batch_accumulation // batch_size if batch_accumulation % batch_size == 0 else batch_accumulation // batch_size + 1
)
batch_accumulation = accumulate_steps * batch_size
total_items = 1367016
total_steps = total_items // batch_accumulation
warmup_steps = total_steps // 25  ### 4% for warmup
lr = 3e-4


train_ds = Dataset_jsonl(tain_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_ds = Dataset_jsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)

train_dl = DataLoader_from_file(train_ds, batch_size=batch_size)
val_dl = DataLoader_from_file(val_ds, batch_size=batch_size)

optim, scheduler = wrapper_db.get_optimizer(warmup_steps=warmup_steps, training_steps=total_steps, lr=lr, wd=0.01)
scheduler.step()
fetch_neighbours = wrapper_db.fetch_neighbours

# %%
# n_iter = 100

losses_train = []
losses_val = []
train_steps = 0

max_val_loss = 10000
freq_val = 1000
num_val = 200
# freq_val = 10
# num_val = 10
### Ensure that validation is performed after taking the gradient step.
freq_val = (freq_val // accumulate_steps) * accumulate_steps

f_train = open(out_folder + "losses_train_optim.txt", "a")
f_val = open(out_folder + "losses_val_optim.txt", "a")
current_time = datetime.now()
text_start = f"\n------- NEW TRAINING {str(current_time)}, batch size = {batch_size}, batch_accum = {batch_accumulation}, warmup steps = {warmup_steps}, validation frequency = {freq_val}, learining rate = {lr}-------\n"
f_train.write(text_start)
f_val.write(text_start)
print(text_start)

tt = time.time()

saved_ind = 0

val_dl_iter = iter(val_dl)

for seq, docs in tqdm(train_dl, total=total_steps):
    seq = seq.cuda()
    retrieved = fetch_neighbours(seq, doc_except=docs)

    train_steps += 1
    loss = retro(seq, retrieved=retrieved, return_loss=True)

    del seq, retrieved
    # gradient step
    loss.backward()

    if train_steps % accumulate_steps == 0:
        optim.step()
        optim.zero_grad()
        scheduler.step()
        losses_train.append(loss.item())
        f_train.write(str(loss.item()) + "\n")
        # del loss

    # if i > n_iter:
    #    break

    losses_val_cur = []

    if train_steps % freq_val == 0:
        retro.eval()
        f_train.flush()
        gc.collect()
        torch.cuda.empty_cache()
        print("------ Validation ------")
        val_step = 0
        for seq, docs in tqdm(val_dl_iter, total=num_val, ncols=80):
            # print(docs)
            val_step += 1
            seq = seq.cuda()
            retrieved = fetch_neighbours(seq, doc_except=docs)

            loss = retro(seq, retrieved=retrieved, return_loss=True)

            losses_val_cur.append(loss.item())
            del loss, seq, retrieved

            if val_step >= num_val:
                break

        if val_step < num_val:
            print("----- Reloading val dataset ------")
            val_ds = Dataset_jsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
            val_dl = DataLoader_from_file(val_ds, batch_size=batch_size)
            val_dl_iter = iter(val_dl)

        if len(losses_val_cur) != 0:
            loss_cur = sum(losses_val_cur) / (len(losses_val_cur))
            losses_val.append(loss_cur)
            f_val.write(str(loss_cur) + "\n")
            f_val.flush()

            print("---- Saving the last model -----")
            model_file_name = model_folder + f"{model_name}_last.pth"
            torch.save(retro.state_dict(), model_file_name)

            if loss_cur < max_val_loss:
                max_val_loss = loss_cur
                print("---- Saving the best model -----")
                model_file_name = model_folder + f"{model_name}_best_{saved_ind}.pth"
                torch.save(retro.state_dict(), model_file_name)
                saved_ind = (saved_ind + 1) % 3

        retro.train()
        gc.collect()
        torch.cuda.empty_cache()

time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")

# %%
