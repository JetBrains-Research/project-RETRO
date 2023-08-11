from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from retro_pytorch.retrieval import get_top_similar
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.training import TrainingWrapper

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
parser.add_argument("-config", "--config", default="config.yaml", help="Config filename")
args = parser.parse_args()
no_retrieve = args.no_retrieve
config_name = args.config

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths
training_params = config.training_params
retrieve_hyperparams = config.retrieve.hyperparams
index_params = config.retrieve.hnsw_params

### TODO rewrite the validation script utilizing pytorch lightning

# Use the arguments in your program
if no_retrieve:
    print("NO retrieve during training")
    add_flag = "_no_retrieve"
else:
    print("Retrieval would be used during training")
    add_flag = ""


add_flag += "_star"
add_flag += "_conc_proj"
knn_path_train = os.path.join(paths.texts_folder, "knn_from_project.dat")
if not no_retrieve:
    config.model_hyperparameters.max_seq_len = 2 * config.model_hyperparameters.max_seq_len
    on_project = True
    print("Training on the retrieval from the projects")

add_flag += "_kimp10"
"""
Training. Add flag --no-retrieve or -no if you want to train without retrieval.
It would add '_no_retrieve' to output filenames (model and train/val loss tracking)
"""

#%%

model_name = paths.model_name + add_flag

train_data_path = os.path.join(paths.data_folder, paths.train_data_file)
val_data_path = os.path.join(paths.data_folder, paths.val_data_file)
stats_path = os.path.join(paths.texts_folder, paths.processed_stats_filename)

with open(stats_path, "r") as f:
    stats = json.load(f)

# config.model_hyperparameters.dec_cross_attn_layers = eval(config.model_hyperparameters.dec_cross_attn_layers)

# import torch
retro = RETRO(**config.model_hyperparameters).cuda()
config.model_hyperparameters.max_seq_len = config.model_hyperparameters.max_seq_len // 2
retro_base = RETRO(**config.model_hyperparameters).cuda()
model_file = paths.model_folder + "retro_star_conc_proj_last_0.pth"
model_file_base = paths.model_folder + "retro_no_retrieve_star_conc_proj_last_1.pth"
retro.load_state_dict(torch.load(model_file))
retro_base.load_state_dict(torch.load(model_file_base))
retro.eval()
retro_base.eval()

#%%

wrapper_db = TrainingWrapper(
    retro=retro,  # path to retro instance
    knn=retrieve_hyperparams.n_knn,  # knn (2 in paper was sufficient)
    chunk_size=config.retrieve.chunk_size,  # chunk size (64 in paper)
    documents_path=paths.data_folder,  # path to folder of text
    data_file_paths=[],
    chunks_memmap_path=os.path.join(paths.texts_folder, "train.chunks.dat"),  # path to chunks
    seqs_memmap_path=os.path.join(paths.texts_folder, "train.seq.dat"),  # path to sequence data
    doc_ids_memmap_path=os.path.join(
        paths.texts_folder, "train.doc_ids.dat"
    ),  # path to document ids per chunk (used for filtering neighbors belonging to same document)
    processed_stats_json_path=stats_path,
    knn_memmap_path=knn_path_train,  # used for the training
    knn_memmap_path_option=None,  # knn_path_optional,  ## used for additional validaton purposes
    split_meta_path=os.path.join(paths.texts_folder, "split_meta_dict.json"),
    knn_extra_neighbors=retrieve_hyperparams.knn_extra_neighbors,  # num extra neighbors to fetch
    precalculate_knn=False,
    index_params=index_params,
)


fetch_self_ret = wrapper_db.fetch_self_ret
fetch_random_chunk = wrapper_db.fetch_random_chunk
generate_pure_random_chunk = wrapper_db.generate_pure_random_chunk
fetch_none = wrapper_db.fetch_none
fetch_previous = wrapper_db.fetch_previous

batch_size_val = training_params.batch_size_val
# batch_size_val = 7
val_dl = iter(wrapper_db.get_dataloader(split="val", return_docs=True, batch_size=batch_size_val, shuffle=False))
# %%

doc_proj_file = os.path.join(paths.texts_folder, "doc_proj_dict.json")
doc_bin_dict_path = os.path.join(paths.texts_folder, "doc_bin_dict.json")
bin_val_losses_path = os.path.join(paths.out_folder, f"bin_val_losses{add_flag}.json")

with open(doc_proj_file, "r") as file:
    doc_proj_dict = json.load(file)

with open(doc_bin_dict_path, "r") as file:
    doc_bin_dict = json.load(file)

doc_proj_dict = {int(key): value for key, value in doc_proj_dict.items()}
doc_bin_dict = {int(key): value for key, value in doc_bin_dict.items()}


def get_batch_from_dict(dict_, keys):
    return [dict_[doc_id] for doc_id in keys]


losses_val: list[list[float]] = []

text_start = f"\n------- FINAL VALIDATION {str(datetime.now())}, batch size = {batch_size_val} -------\n"
print(text_start)

bin_losses_dict = defaultdict(list)
bin_losses_dict["step"] = 0
bin_losses_dict[
    "messeage"
] = f"Started at {str(datetime.now())}, Batch size = {batch_size_val}, Retrieve = {not no_retrieve}"
return_itemwise = True
tt = time.time()

with torch.no_grad():
    for step, (seq, ret, docs) in enumerate(tqdm(val_dl, ncols=80), start=1):
        seq = seq.cuda()
        bins = get_batch_from_dict(doc_bin_dict, docs.numpy())

        losses_cur = []

        if no_retrieve:
            loss = retro(seq, retrieved=None, return_loss=True, return_itemwise=return_itemwise)
            losses_cur = [loss.cpu()]
        else:
            seq_cut = seq[1:]

            for fetch_neighbours in [
                fetch_self_ret
            ]:  # , fetch_random_chunk, generate_pure_random_chunk, fetch_previous
                retrieved = fetch_neighbours(seq, ret=ret)
                if fetch_neighbours != fetch_previous:
                    retrieved = retrieved[1:]

                (b, seq_size, n_knn, chun_dsize) = ret.shape

                # retrieved = rearrange(retrieved[:, :, 0, : chun_dsize // 2], "b s c -> b (s c)")
                retrieved = rearrange(retrieved, "b s k c -> b (s k c)")
                retrieved = get_top_similar(retrieved=retrieved, context=seq_cut, k_imp=128, pad_id=0)
                retrieved = rearrange(retrieved, "b (s c) -> b s 1 c", c=chun_dsize // 2)
                retrieved = torch.cat((retrieved, retrieved), dim=-1)

                loss_and_recall = retro(
                    seq_cut.cuda(),
                    retrieved=retrieved.cuda(),
                    return_loss=True,
                    return_itemwise=return_itemwise,
                    return_recall=True,
                    k_list=[1, 3, 5],
                )
                losses_cur.append(loss_and_recall)

            # loss_and_recall = retro_base(seq_cut.cuda(), retrieved=None, return_loss=True, return_itemwise=return_itemwise,
            #              return_recall = True, k_list = [1, 3, 5])
            # losses_cur.append(loss_and_recall)

        losses = torch.stack(losses_cur, dim=1)
        for bin, loss in zip(bins, losses):
            bin_losses_dict[bin].append(loss.tolist())

        # if step >= 30:
        #     break

        if step % 120 == 0:
            bin_losses_dict["step"] = step
            with open(bin_val_losses_path, "w") as f:
                json.dump(bin_losses_dict, f)

bin_losses_dict["step"] = step
with open(bin_val_losses_path, "w") as f:
    json.dump(bin_losses_dict, f)

time_used = time.time() - tt
print(f"Time used = {time_used:.2f} s")

#%%
