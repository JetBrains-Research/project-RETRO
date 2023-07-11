import torch

from retro_pytorch.utils import seed_all

seed_all(1111)

import argparse
import gc
import time
import os

from omegaconf import OmegaConf

from retro_pytorch.dataloaders import DataLoaderFromFile, DatasetJsonl
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.train_functions import val_steps
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
no_retrieve = args.no_retrieve
config_name = args.config

# # loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(**config.model_hyperparameters).cuda()

#%%

val_data_path = os.path.join(paths.data_folder, paths.val_data_file)

gc.collect()
torch.cuda.empty_cache()

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
    processed_stats_json_path=os.path.join(paths.texts_folder + "processed-stats.json"),
    # max_chunks = n_chuncks,                        # maximum cap to chunks
    # max_seqs = n_chuncks//5,                            # maximum seqs
    knn_extra_neighbors=100,  # num extra neighbors to fetch
    max_index_memory_usage="10G",
    current_memory_available="32G",
)

#%%

# def random_seq():
#     torch.randint(0, 30000)

batch_size = 16
total_items = 136701

print(f"Batch size = {batch_size}")

val_ds = DatasetJsonl(val_data_path, cnunk_size=64, seq_length=512, pad_id=0)
val_dl = DataLoaderFromFile(val_ds, batch_size=batch_size)
val_dl_iter = iter(val_dl)

fetch_neighbours = wrapper_db.fetch_neighbours
losses_val: list[float] = []

# model_file = model_folder + 'retro_no_retrieve_last.pth'
model_file = paths.model_folder + "retro_best_0.pth"
retro.load_state_dict(torch.load(model_file))
retro.eval()

tt = time.time()  ## TODO rewrite this procedure to match new trainig pipeline
losses_val, _ = val_steps(retro, no_retrieve, fetch_neighbours, num_val=100, val_dl_iter=val_dl_iter)

val_avg = sum(losses_val) / len(losses_val)
print(f"Average validation loss = {val_avg}")
time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")
