from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import jsonlines
from collections import defaultdict
import os
from tqdm import tqdm

from omegaconf import OmegaConf
import numpy as np

parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
stats_path = os.path.join(paths.texts_folder, "processed-stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)
# %%

splits = ['val', 'test', 'train']

split_doc_dict = defaultdict(list)

for split in splits:
    
    file = os.path.join(paths.data_folder, split + ".jsonl")

    with jsonlines.open(file) as reader:
        for line in tqdm(reader):
            doc_id = line["doc_id"]
            split_doc_dict[split].append(doc_id)

for key in split_doc_dict:
    split_doc_dict[key].sort()

with open(paths.data_folder + "split_doc_dict.json", "w") as file:
    json.dump(split_doc_dict, file)

# %%

doc_ids_memmap_path = os.path.join(paths.texts_folder, "train.doc_ids.dat")
doc_ids = np.memmap(doc_ids_memmap_path, dtype=np.int32, mode="r")
doc_ids = np.array(doc_ids)

# seqs_memmap - contains chunk ids of the beggining of each sequence.
seq_memmap_path = os.path.join(paths.texts_folder, "train.seq.dat")
seq_ids = np.memmap(seq_memmap_path, dtype=np.int32, mode="r")
seq_ids = np.array(seq_ids)

## crop zero elemenst in seq_ids
last_non_zero_index = np.where(seq_ids != 0)[0][-1] + 1
seq_ids = seq_ids[:last_non_zero_index]

# seq_docs - contains doc ids of the beggining of each sequence.
seq_docs = doc_ids[seq_ids]
#%%

split_meta_dict = dict()

for split in splits:
    first_doc = split_doc_dict[split][0]
    last_doc = split_doc_dict[split][-1]
    mask = (seq_docs >= first_doc) & (seq_docs <= last_doc)
    num_seqs = np.sum(mask)
    first_seq_ind = np.where(seq_docs == first_doc)[0][0]
    split_meta_dict[split] = {'split size in seqs':int(num_seqs), 'first doc_id': int(first_doc), 'first sequence index': int(first_seq_ind)}


with open(paths.data_folder + "split_meta_dict.json", "w") as file:
    json.dump(split_meta_dict, file)

#%%

with open(paths.data_folder + "split_meta_dict.json", "r") as file:
    tst = json.load(file)

#%%
# with open(paths.data_folder + "split_doc_dict.json", "r") as file:
#     split_doc_dict = json.load(file)

# split_ind_dict = defaultdict(list)

# for split in splits:
#     num_chunks = stats['chunks']
#     doc_ids_memmap_path = os.path.join(paths.texts_folder, "train.doc_ids.dat")
#     doc_ids = np.memmap(doc_ids_memmap_path, dtype=np.int32, mode="r")
#     doc_ids = np.array(doc_ids)
    
#     split_docs = np.array(split_doc_dict[split])
    
#     mask = np.isin(doc_ids, split_docs)
#     np.sum(mask)
#     indices_in_split_docs = np.where(mask)[0]
#     split_ind_dict[split] = indices_in_split_docs.tolist()

# with open(paths.data_folder + "split_ind_dict.json", "w") as file:
#     json.dump(split_ind_dict, file)
# %%

