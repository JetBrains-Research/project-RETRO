from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import os
import time
from datetime import datetime

from omegaconf import OmegaConf
from tqdm import tqdm

from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.train_functions import grad_step, val_steps, val_update
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
no_retrieve = args.no_retrieve
config_name = args.config

# Use the arguments in your program
if no_retrieve:
    print("NO retrieve during training")
    add_flag = "_no_retrieve"
else:
    print("Retrieval would be used during training")
    add_flag = ""

"""
Training. Add flag --no-retrieve or -no if you want to train without retrieval.
It would add '_no_retrieve' to output filenames (model and train/val loss tracking)
"""

#%%

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
training_params = config.training_params
retrieve_hyperparams = config.retrieve.hyperparams
index_params = config.retrieve.hnsw_params

model_name = paths.model_name + add_flag

train_data_path = os.path.join(paths.data_folder, paths.train_data_file)
val_data_path = os.path.join(paths.data_folder, paths.val_data_file)
filename_train = os.path.join(paths.out_folder, paths.out_filename_train + add_flag + ".txt")
filename_val = os.path.join(paths.out_folder, paths.out_filename_val + add_flag + ".txt")
stats_path = os.path.join(paths.texts_folder, "processed-stats.json")
f_train = open(filename_train, "a")
f_val = open(filename_val, "a")
with open(stats_path, "r") as f:
    stats = json.load(f)

on_project = True
if on_project:
    print('Training on the retrieval from the projects')
    add_flag = '_conc_proj'
    knn_path_train=os.path.join(paths.texts_folder, 'knn_per_project.dat'),
    knn_path_optional=os.path.join(paths.texts_folder, 'knn_from_all.dat'),
else:
    print('Training on the retrieval from the all dataset')
    add_flag = '_conc_all'
    knn_path_train=os.path.join(paths.texts_folder, 'knn_from_all.dat'),
    knn_path_optional=os.path.join(paths.texts_folder, 'knn_per_project.dat'),

#config.model_hyperparameters.dec_cross_attn_layers = eval(config.model_hyperparameters.dec_cross_attn_layers)

# instantiate RETRO, fit it into the TrainingWrapper with correct settings
# import torch
config.model_hyperparameters.max_seq_len += 512
retro = RETRO(**config.model_hyperparameters).cuda()
# model_file = paths.model_folder + "retro_concat_last_1epoch.pth"
# retro.load_state_dict(torch.load(model_file))
retro.train()

print("Freezing encoder parameters")
for param in retro.encoder.parameters():
    param.requires_grad = False

with open(paths.data_folder + "split_doc_dict.json", "r") as file:
    split_doc_dict = json.load(file)
with open(paths.data_folder + "split_ind_dict.json", "r") as file:
    split_ind_dict = json.load(file)

split_len_dic = {key: len(value) for key, value in split_ind_dict.items()}
#%%

wrapper_db = TrainingWrapper(
    retro=retro,  # path to retro instance
    knn=retrieve_hyperparams.n_knn,  # knn (2 in paper was sufficient)
    chunk_size=stats["chunk_size"],  # chunk size (64 in paper)
    documents_path=paths.data_folder,  # path to folder of text
    data_file_paths=[],
    chunks_memmap_path=os.path.join(paths.texts_folder, "train.chunks.dat"),  # path to chunks
    seqs_memmap_path=os.path.join(paths.texts_folder, "train.seq.dat"),  # path to sequence data
    doc_ids_memmap_path=os.path.join(
        paths.texts_folder, "train.doc_ids.dat"
    ),  # path to document ids per chunk (used for filtering neighbors belonging to same document)
    processed_stats_json_path=stats_path,
    knn_memmap_path=knn_path_train, # used for the training
    knn_memmap_path_option=knn_path_optional, ## used for additional validaton purposes
    split_meta_path =  os.path.join(paths.texts_folder, 'split_meta_dict.json'),
    knn_extra_neighbors=retrieve_hyperparams.knn_extra_neighbors,  # num extra neighbors to fetch
    precalculate_knn=False,
    index_params=index_params,
)

fetch_random_chunk = wrapper_db.fetch_random_chunk
generate_pure_random_chunk = wrapper_db.generate_pure_random_chunk

### setting up number of steps.
freq_val = training_params.freq_val  # frequency of validation
num_val = training_params.num_val  # number of validation steps
batch_size = training_params.batch_size
batch_size_val = training_params.batch_size_val
batch_accumulation = training_params.batch_accumulation
total_items = 1367016

accumulate_steps = (
    batch_accumulation // batch_size if batch_accumulation % batch_size == 0 else batch_accumulation // batch_size + 1
)
batch_accumulation = accumulate_steps * batch_size
total_steps = total_items // batch_accumulation
warmup_steps = total_steps // 25  ### 4% for warmup
lr = training_params.lr  # learning rate

### Ensure that validation is performed after taking the gradient step.
freq_val = (freq_val // accumulate_steps) * accumulate_steps

#%%
val_dl = iter(wrapper_db.get_dataloader(split='val', batch_size=batch_size_val, shuffle = True))

optim, scheduler = wrapper_db.get_optimizer(warmup_steps=warmup_steps, training_steps=total_steps, lr=lr, wd=0.01)
scheduler.step()
retro.train()
# %%

losses_train: list[float] = []
losses_val: list[list[float]] = []
losses_val_pure_rnd: list[float] = []
losses_val_chunk_rnd: list[float] = []
train_steps = 0
max_val_loss = 10000.0

text_start = f"\n------- NEW TRAINING {str(datetime.now())}, batch size = {batch_size}, batch_accum = {batch_accumulation}, warmup steps = {warmup_steps}, validation frequency = {freq_val}, learining rate = {lr}-------\n"
f_train.write(text_start)
f_val.write(text_start)
print(text_start)

tt = time.time()

saved_ind = 0
saved_last_ind = 0
for epoch in range(2):
    train_dl = iter(wrapper_db.get_dataloader(split='train', batch_size=batch_size, shuffle=True, num_workers=4))
    print(f'---------  EPOCH {epoch} ---------')
    for train_steps, (seq, ret1, ret2) in enumerate(tqdm(train_dl, total=total_items // batch_size), start=1):

        if no_retrieve:
            ret1 = None
        loss = retro(seq.cuda(), retrieved=ret1.cuda(), return_loss=True)
        loss.backward()

        if train_steps % accumulate_steps == 0:
            grad_step(optim, scheduler, loss, losses_train, f_train)
            f_train.flush()

        if train_steps % freq_val == 0:

            f_train.flush()
            retro.eval()

            print("------ Validation ------")

            losses_val_cur, val_step = val_steps(retro, val_dl, num_val=num_val, no_retrieve = no_retrieve,
                                           fetch_neighbours_list = [fetch_random_chunk, generate_pure_random_chunk])
            if val_step < num_val:
                print("Reloading VAL Dataloader")
                val_dl = iter(wrapper_db.get_dataloader(split='val', batch_size=batch_size_val, shuffle = True))

            # if no_retrieve:
            #     losses_val_rnd_cur = losses_val_cur
            #     losses_val_pure_rnd_cur = losses_val_cur

            max_val_loss, saved_ind, saved_last_ind = val_update(
                retro,
                losses_val,
                losses_val_cur,
                paths.model_folder,
                model_name,
                f_val,
                max_val_loss,
                saved_ind,
                saved_last_ind,
            )

            retro.train()

time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")

#%%