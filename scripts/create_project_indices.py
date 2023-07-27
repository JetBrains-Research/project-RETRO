from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import os
import time

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from retro_pytorch.retrieval import MODEL_DIM, build_compound_index
from retro_pytorch.utils import memmap

#%%

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
stats_path = os.path.join(paths.texts_folder, "processed-stats.json")
with open(stats_path, "r") as f:
    stats = json.load(f)

with open(paths.data_folder + "proj_doc_dict.json", "r") as file:
    proj_doc_dict = json.load(file)
#%%

# instantiate RETRO, fit it into the TrainingWrapper with correct settings
num_chunks = stats["chunks"]
num_nearest_neighbors = retrieve_hyperparams.n_knn

chunks_memmap_path = os.path.join(paths.texts_folder, "train.chunks.dat")
doc_ids_memmap_path = os.path.join(paths.texts_folder, "train.doc_ids.dat")
embedding_path = f"{chunks_memmap_path}.embedded"
knn_path = os.path.join(paths.texts_folder, "knn_per_project.dat")

embed_shape = (num_chunks, MODEL_DIM)

doc_ids = np.memmap(doc_ids_memmap_path, shape=(num_chunks,), dtype=np.int32, mode="r")
doc_ids = np.array(doc_ids)
embeddings_all = np.memmap(embedding_path, shape=embed_shape, dtype=np.float32, mode="r")
#%%


def calculate_per_project_knn(
    embeddings_all,
    index_params,
    knn_path,
    proj_doc_dict,
    num_chunks,
    num_nearest_neighbors=2,
    num_extra_neighbours=100,
    max_iter=-1,
):

    error_rates = []
    tt = time.time()
    model_dim = embeddings_all.shape[1]

    with memmap(knn_path, shape=(num_chunks, num_nearest_neighbors), dtype=np.int32, mode="w+") as knns:

        for proj_id in tqdm(proj_doc_dict.keys()):

            doc_list = np.array(proj_doc_dict[str(proj_id)])
            indices = np.where(np.isin(doc_ids, doc_list))[0]
            query_doc_ids = doc_ids[indices]
            embeddings = embeddings_all[indices]
            index = build_compound_index(
                embeddings, index_file="", index_params=index_params, d=model_dim, verbose=False, save_to_file=False
            )

            ## ensures that there no retrieve from the same document
            for n_extra in [num_extra_neighbours, 3 * num_extra_neighbours, 10 * num_extra_neighbours]:

                dist, ind = index.search(embeddings, k=num_nearest_neighbors + n_extra)
                l = max(len(dist), 1)
                error_rate = sum(dist[:, 0]) / l
                error_rates.append([proj_id, error_rate])
                dist = dist[:, 1:]
                ind = ind[:, 1:]

                doc_ids_selected = query_doc_ids[ind]
                neighbor_from_same_doc = query_doc_ids[..., None] == doc_ids_selected
                ind = np.where(neighbor_from_same_doc, -1, ind)
                dist = np.where(neighbor_from_same_doc, 1e3, dist)
                ind = np.take_along_axis(ind, np.argsort(dist, axis=1), axis=1)

                ind = ind[:, :num_nearest_neighbors]

                doc_ids_selected = query_doc_ids[ind]
                neighbor_from_same_doc = query_doc_ids[..., None] == doc_ids_selected
                if np.sum(neighbor_from_same_doc) == 0:
                    break

            if np.sum(neighbor_from_same_doc) > 0:
                print(f"Retrieve from the same doc!, project id = {proj_id}")
            indices_selected = indices[ind]

            knns[indices] = indices_selected

            # if max_iter > 0 and step > max_iter:
            #     break

    print(f"KNNs are saved into {knn_path}")
    print(f"Time used = {(time.time() - tt):.2f}")

    error_rates = np.array(error_rates)[:, 1].astype(float)
    error_rates_av = np.mean(error_rates[error_rates > 0])

    print(f"Number of nonzero error rates  = {np.sum(error_rates>0)}")
    print(f"Average nonzero error rate  = {error_rates_av}")
    print(f"Max error rate  = {np.max(error_rates)}")


#%%

calculate_per_project_knn(embeddings_all, index_params, knn_path, proj_doc_dict, num_chunks, max_iter=-1)

#%%

knn_map = np.memmap(knn_path, shape=(num_chunks, num_nearest_neighbors), dtype=np.int32, mode="r")
knn_map = np.array(knn_map)

n_samples = 50_000
random_ind = np.random.randint(num_chunks, size=n_samples)

print(f"Testing on {n_samples} samples")

neighb_ind = knn_map[random_ind]
mask = np.any(neighb_ind != [0, 0], axis=1)
neighb_ind = neighb_ind[mask]
if not len(neighb_ind) == n_samples:
    print("Some indices are missing")
emb_query = embeddings_all[random_ind][mask]
neighb_emb = embeddings_all[neighb_ind]
neighb_emb_wrong = embeddings_all[neighb_ind - 30_000]

dist_good = np.linalg.norm(neighb_emb - emb_query[:, np.newaxis, :], axis=-1)
dist_wrong = np.linalg.norm(neighb_emb_wrong - emb_query[:, np.newaxis, :], axis=-1)

#%%

dist_good_1 = dist_good[:, 0]
dist_good_2 = dist_good[:, 1]
dist_wrong = dist_wrong[:, 0]

mean_1 = np.mean(dist_good_1[dist_good_1 > 0])
mean_2 = np.mean(dist_good_2[dist_good_2 > 0])
mean_wrong = np.mean(dist_wrong[dist_good_1 > 0])
std_1 = np.std(dist_good_1[dist_good_1 > 0])
std_2 = np.std(dist_good_2[dist_good_2 > 0])
std_wrong = np.std(dist_wrong[dist_good_1 > 0])


print(f"Mean distance for best neighbours        {mean_1:.2f} +- {std_1:.2f}")
print(f"Mean distance for second best neighbours {mean_2:.2f} +- {std_2:.2f}")
print(f"Mean distance for wrong samples          {mean_wrong:.2f} +- {std_wrong:.2f}")
