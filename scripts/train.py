from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import time
from datetime import datetime

from omegaconf import OmegaConf
from tqdm import tqdm

from retro_pytorch.dataloaders import DataLoaderFromFile, DatasetJsonl
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.train_functions import calc_loss, grad_step, val_steps, val_upadate
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
args = parser.parse_args()
no_retrieve = args.no_retrieve

# Use the arguments in your program
if no_retrieve:
    print(f"NO retrieve during training")
    add_flag = "_no_retrieve"
else:
    print(f"Retrieval would be used during training")
    add_flag = ""

"""
Training. Add flag --no-retrieve or -no if you want to train without retrieval.
It would add '_no_retrieve' to output filenames (model and train/val loss tracking)
"""

# # loading pathes
conf_load = OmegaConf.load("config.yaml")
paths = conf_load["paths"]

model_name = paths.model_name + add_flag
tain_data_path = paths.data_folder + paths.tain_data_file
val_data_path = paths.data_folder + paths.val_data_file
filename_train = paths.out_folder + paths.out_filename_train + add_flag + ".txt"
filename_val = paths.out_folder + paths.out_filename_val + add_flag + ".txt"
f_train = open(filename_train, "a")
f_val = open(filename_val, "a")

## TODO I am not sure that at this step I want to move model hyperparameters to a config file - it is easier for me to read it here.
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

wrapper_db = TrainingWrapper(
    retro=retro,  # path to retro instance
    knn=2,  # knn (2 in paper was sufficient)
    chunk_size=64,  # chunk size (64 in paper)
    documents_path=paths.data_folder,  # path to folder of text
    data_file_paths=[],
    chunks_memmap_path=paths.texts_folder + "train.chunks.dat",  # path to chunks
    seqs_memmap_path=paths.texts_folder + "train.seq.dat",  # path to sequence data
    doc_ids_memmap_path=paths.texts_folder
    + "train.doc_ids.dat",  # path to document ids per chunk (used for filtering neighbors belonging to same document)
    processed_stats_json_path=paths.texts_folder + "processed-stats.json",
    # max_chunks = n_chuncks,                        # maximum cap to chunks
    # max_seqs = n_chuncks//5,                            # maximum seqs
    knn_extra_neighbors=100,  # num extra neighbors to fetch
    max_index_memory_usage="10G",
    current_memory_available="32G",
)

# %%

### setting up number of steps. freq_val - frequency of validation, num_val - number of validation steps

freq_val = 6
num_val = 6
batch_size = 2
batch_accumulation = 6  # 64
total_items = 1367016

accumulate_steps = accumulate_steps = (
    batch_accumulation // batch_size if batch_accumulation % batch_size == 0 else batch_accumulation // batch_size + 1
)
batch_accumulation = accumulate_steps * batch_size
total_steps = total_items // batch_accumulation
warmup_steps = total_steps // 25  ### 4% for warmup
lr = 3e-4

### Ensure that validation is performed after taking the gradient step.
freq_val = (freq_val // accumulate_steps) * accumulate_steps

# loading data and optimization functions.
train_ds = DatasetJsonl(tain_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
train_dl = DataLoaderFromFile(train_ds, batch_size=batch_size)
val_dl = DataLoaderFromFile(val_ds, batch_size=batch_size)

optim, scheduler = wrapper_db.get_optimizer(warmup_steps=warmup_steps, training_steps=total_steps, lr=lr, wd=0.01)
scheduler.step()
fetch_neighbours = wrapper_db.fetch_neighbours

# %%

losses_train: list[float] = []
losses_val: list[float] = []
train_steps = 0
max_val_loss = 10000

text_start = f"\n------- NEW TRAINING {str(datetime.now())}, batch size = {batch_size}, batch_accum = {batch_accumulation}, warmup steps = {warmup_steps}, validation frequency = {freq_val}, learining rate = {lr}-------\n"
f_train.write(text_start)
f_val.write(text_start)
print(text_start)

tt = time.time()

saved_ind = 0
val_dl_iter = iter(val_dl)

for train_steps, (seq, docs) in enumerate(tqdm(train_dl, total=total_steps), start=1):

    loss = calc_loss(seq, docs, retro, no_retrieve, fetch_neighbours)

    if train_steps % accumulate_steps == 0:
        grad_step(optim, scheduler, loss, losses_train, f_train)

    if train_steps % freq_val == 0:

        f_train.flush()
        losses_val_cur, val_step = val_steps(retro, no_retrieve, fetch_neighbours, num_val, val_dl_iter)
        max_val_loss, saved_ind, val_dl_iter = val_upadate(
            retro,
            losses_val,
            losses_val_cur,
            paths.model_folder,
            model_name,
            val_dl_iter,
            f_val,
            max_val_loss,
            saved_ind,
        )

        if val_step < num_val:
            print("----- Reloading val dataset ------")
            val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
            val_dl = DataLoaderFromFile(val_ds, batch_size=batch_size)
            val_dl_iter = iter(val_dl)

        retro.train()

time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")

# %%
