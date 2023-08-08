from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import os

import numpy as np
from omegaconf import OmegaConf

from retro_pytorch.retrieval import calculate_per_project_knn, test_knn

parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config

"""
Training. Add flag --no-retrieve or -no if you want to train without retrieval.
It would add '_no_retrieve' to output filenames (model and train/val loss tracking)
"""

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
training_params = config.training_params
retrieve_hyperparams = config.retrieve.hyperparams
index_params = config.retrieve.hnsw_params

train_data_path = os.path.join(paths.data_folder, paths.train_data_file)
val_data_path = os.path.join(paths.data_folder, paths.val_data_file)
stats_path = os.path.join(paths.texts_folder, paths.processed_stats_filename)
with open(stats_path, "r") as f:
    stats = json.load(f)

with open(paths.texts_folder + "proj_doc_dict.json", "r") as file:
    proj_doc_dict = json.load(file)
#%%

# instantiate RETRO, fit it into the TrainingWrapper with correct settings
num_chunks = stats["chunks"]
num_nearest_neighbors = retrieve_hyperparams.n_knn

chunks_memmap_path = os.path.join(paths.texts_folder, "train.chunks.dat")
doc_ids_memmap_path = os.path.join(paths.texts_folder, "train.doc_ids.dat")
embedding_path = f"{chunks_memmap_path}.embedded"
knn_path = os.path.join(paths.texts_folder, "knn_per_project.dat")


calculate_per_project_knn(doc_ids_memmap_path, embedding_path, index_params, knn_path, proj_doc_dict, num_chunks)

test_knn(embedding_path, knn_path, num_chunks, num_nearest_neighbors=2, n_samples=50_000)
