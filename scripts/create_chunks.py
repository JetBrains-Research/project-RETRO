from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import gc
import json
import os
import time

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

n_chuncks = 20_000_000
# n_chuncks = 700_000

texts_folder = paths.texts_folder
retrieve_hyperparams = config.retrieve.hyperparams
index_params = config.retrieve.hnsw_params
stats_path = os.path.join(paths.texts_folder, paths.processed_stats_filename)

gc.collect()
torch.cuda.empty_cache()

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(**config.model_hyperparameters).cuda()

tt = time.time()

wrapper = TrainingWrapper(
    retro=retro,  # path to retro instance
    knn=retrieve_hyperparams.n_knn,  # knn (2 in paper was sufficient)
    chunk_size=config.retrieve.chunk_size,  # chunk size (64 in paper)
    documents_path=paths.data_folder,  # path to folder of text
    data_file_paths=[
        os.path.join(paths.data_folder, "train.parquet"),
        os.path.join(paths.data_folder, "val.parquet"),
        # os.path.join(paths.data_folder, "test.jsonl"),
        # os.path.join(paths.data_folder, "test_dev.parquet"),
    ],
    chunks_memmap_path=os.path.join(texts_folder, "train.chunks.dat"),  # path to chunks
    seqs_memmap_path=os.path.join(texts_folder, "train.seq.dat"),  # path to sequence data
    doc_ids_memmap_path=os.path.join(
        texts_folder, "train.doc_ids.dat"
    ),  # path to document ids per chunk (used for filtering neighbors belonging to same document)
    knn_memmap_path=os.path.join(texts_folder, "knn_from_project.dat"),
    processed_stats_json_path=stats_path,
    split_meta_path=os.path.join(paths.texts_folder, "split_meta_dict.json"),
    proj_doc_dict_path=os.path.join(paths.texts_folder, "proj_doc_dict.json"),
    doc_proj_dict_path = os.path.join(paths.texts_folder, 'doc_proj_dict.json'),
    doc_bin_dict_path = os.path.join(paths.texts_folder, 'doc_bin_dict_path'),
    max_chunks=n_chuncks,  # maximum cap to chunks
    max_seqs=n_chuncks // 5,  # maximum seqs
    knn_extra_neighbors=retrieve_hyperparams.knn_extra_neighbors,  # num extra neighbors to fetch
    precalculate_knn=False,
    index_params=index_params,
)

time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")
