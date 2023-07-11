import argparse
import json
import time
from typing import Any

import faiss
import numpy as np
from omegaconf import OmegaConf
import os

# %%

"""
Tests precision of the index search for a given pre-calculated embeddings file
"""

parser = argparse.ArgumentParser(description="")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
config_name = args.config

print(f"Loading configs from {config_name} file")
conf_load = OmegaConf.load(config_name)
paths = conf_load.paths

stats = json.load(open(os.path.join(paths.texts_folder, "processed-stats.json")))
num_chunks = stats["chunks"]
# index_info = json.load(open(os.path.join(paths.texts_folder, "index_infos.json")))

#%%


def faiss_read_index(path: str) -> Any:
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)


tt = time.time()
print(f"loading embeddings from {paths.texts_folder}train.chunks.dat.embedded")
embeddings = np.memmap(
    os.path.join(paths.texts_folder, "train.chunks.dat.embedded"), dtype=np.float32, mode="r", shape=(num_chunks, 768)
)

# embeddings = np.array(embeddings)
print(f"loading index from {paths.texts_folder}knn.index")
faiss_index = faiss_read_index(os.path.join(paths.texts_folder, "knn.index"))

delta = 0.1

for i in range(1):

    all_ind = np.arange(num_chunks)
    n_tot = 30_000
    rand_ind = np.random.choice(all_ind, n_tot, replace=False)
    # rand_ind = np.arange(n_tot)+i*1000_000
    print("ON RANDOM")
    print(f"Started test on {100*n_tot/num_chunks:.2f}% of the data")
    dist, indices_res = faiss_index.search(embeddings[rand_ind], k=1)
    dist = dist[:, 0]
    indices_res = indices_res[:, 0]
    print(f"Retrival results checked {n_tot}")
    print(f"Errors rate = {sum(dist != 0)/n_tot}")
    print(f"Duplication rate = {np.count_nonzero(indices_res - all_ind[rand_ind])/n_tot}")

    dist, indices_res = faiss_index.search(embeddings[rand_ind] + delta, k=1)
    dist = dist[:, 0]
    indices_res = indices_res[:, 0]
    print(f"Noise error rate {sum((dist - (768)*delta**2)>1e-4)/n_tot}")

    #%%

    dist_err = dist[dist != 0]
    print(f"Mean distance error = {np.mean(dist_err**0.5):.2f}")
    time_used = time.time() - tt
    print(f"Time used = {time_used:.2f} s")
