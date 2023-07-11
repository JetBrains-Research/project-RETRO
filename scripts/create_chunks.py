from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import gc
import time
import os

import torch
from omegaconf import OmegaConf

from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config

print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths


"""
Create your chunks and chunk start indices (for calculating sequence ranges for autoregressive training) using text_folder_to_chunks_
Creates embeddings and finding knns for each chuncks in dataset
"""

n_chuncks = 15_000_000
# n_chuncks = 700_000

texts_folder = paths.texts_folder

gc.collect()
torch.cuda.empty_cache()

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(**config.model_hyperparameters).cuda()

tt = time.time()

wrapper = TrainingWrapper(
    retro=retro,  # path to retro instance
    knn=2,  # knn (2 in paper was sufficient)
    chunk_size=64,  # chunk size (64 in paper)
    documents_path=paths.data_folder,  # path to folder of text
    # glob = '**/*.txt',                             # text glob
    data_file_paths=[
        os.path.join(paths.data_folder, "val.jsonl"),
        # os.path.join(paths.data_folder, "test.jsonl"),
        # os.path.join(paths.data_folder, "train.jsonl"),
    ],
    chunks_memmap_path=os.path.join(texts_folder, "train.chunks.dat"),  # path to chunks
    seqs_memmap_path=os.path.join(texts_folder, "train.seq.dat"),  # path to sequence data
    doc_ids_memmap_path=os.path.join(texts_folder, "train.doc_ids.dat"),  # path to document ids per chunk (used for filtering neighbors belonging to same document)
    processed_stats_json_path=os.path.join(texts_folder, "processed-stats.json"),
    max_chunks=n_chuncks,  # maximum cap to chunks
    max_seqs=n_chuncks // 5,  # maximum seqs
    knn_extra_neighbors=100,  # num extra neighbors to fetch
)

time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")
