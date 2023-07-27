from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import os
import time
from datetime import datetime

import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from retro_pytorch.retrieval import SOS_ID
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.train_functions import aggregate_batches, grad_step, val_steps, val_update
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config

#%%

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
training_params = config.training_params
retrieve_hyperparams = config.retrieve.hyperparams
index_params = config.retrieve.hnsw_params


train_data_path = os.path.join(paths.data_folder, paths.train_data_file)
val_data_path = os.path.join(paths.data_folder, paths.val_data_file)
stats_path = os.path.join(paths.texts_folder, "processed-stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(**config.model_hyperparameters).cuda()


with open(paths.data_folder + "split_doc_dict.json", "r") as file:
    split_doc_dict = json.load(file)
with open(paths.data_folder + "split_ind_dict.json", "r") as file:
    split_ind_dict = json.load(file)
with open(paths.data_folder + "proj_doc_dict.json", "r") as file:
    proj_doc_dict = json.load(file)

doc_proj_dict = dict()
for project_id, doc_ids in proj_doc_dict.items():
    for doc_id in doc_ids:
        doc_proj_dict[doc_id] = project_id
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
    knn_memmap_path=os.path.join(paths.texts_folder, "knn_per_project.dat"),
    knn_memmap_path_option=os.path.join(paths.texts_folder, "knn_from_all.dat"),
    split_meta_path=os.path.join(paths.texts_folder, "split_meta_dict.json"),
    knn_extra_neighbors=retrieve_hyperparams.knn_extra_neighbors,  # num extra neighbors to fetch
    precalculate_knn=False,
    index_params=index_params,
)

# %%
batch_size = 10000
split = "train"
dl = iter(wrapper_db.get_dataloader(split=split, batch_size=batch_size, shuffle=True, num_workers=8))
docs_split = set(split_doc_dict[split])
# %%

print(f"Testing splitting the docs on {split} split")

docs_list = []

for train_steps, (seq, ret1, ret2, docs_chunks, docs_knns) in enumerate(tqdm(dl), start=1):
    docs_list.append(torch.flatten(docs_chunks))

docs_list = torch.cat(docs_list)
docs_list = set(docs_list.tolist())
#%%

if docs_list == docs_split:
    print(f"{split} split is ok!")
else:
    print("someting wrong")

#%%


"""
Tests that DL generates chunks from correct project
"""

print(f"Testing correctness of KNN on {split} split")
dl = iter(wrapper_db.get_dataloader(split=split, batch_size=batch_size, shuffle=True, num_workers=8))
same_docs = []
diff_projects = []
all_docs_list = []

for train_steps, (seq, ret1, ret2, docs_chunks, docs_knns) in enumerate(tqdm(dl), start=1):

    same_docs_n = torch.sum(docs_chunks == docs_knns[:, :, 0]) + torch.sum(docs_chunks == docs_knns[:, :, 1])
    all_docs = torch.sum(docs_chunks != docs_knns[:, :, 0]) + torch.sum(docs_chunks != docs_knns[:, :, 1])
    same_docs.append(same_docs_n.item())
    all_docs_list.append(all_docs.item())

    docs_chunks.apply_(lambda element: int(doc_proj_dict[element]))
    docs_knns.apply_(lambda element: int(doc_proj_dict[element]))

    diff_proj_n = torch.sum(docs_chunks != docs_knns[:, :, 0]) + torch.sum(docs_chunks != docs_knns[:, :, 1])
    diff_projects.append(diff_proj_n.item())

    # if train_steps > 10:
    #     break
# %%

same_docs = sum(same_docs)
diff_projects = sum(diff_projects)

print(f"Counted all chunks = {sum(all_docs_list)}")

if same_docs == 0:
    print(f"Split {split}. All retrieved are from different docs than query")
else:
    print(f"Split {split}. Number of same-docs chunks = {same_docs}")

if diff_projects == 0:
    print(f"Split {split}. All retrieved are from same project as query")
else:
    print(f"Split {split}. Number of same-docs chunks = {diff_projects}")
