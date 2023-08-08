import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch import einsum
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from retro_pytorch.utils import memmap, reset_folder_

### TODO I do not know how to pass config name here from the main script

config = OmegaConf.load("config.yaml")
paths = config.paths

# helper functions


def exists(val):
    return val is not None


def range_chunked(max_value, *, batch_size):
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)
        counter = curr


# indexing helper functions


def faiss_read_index(path):
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)


# singleton globals

print(config.messages.embedder_init)
TOKENIZER = AutoTokenizer.from_pretrained(paths.encoder_path)
MODEL = T5ForConditionalGeneration.from_pretrained(paths.encoder_path).base_model.encoder
MODEL = MODEL.cuda()
MODEL.eval()

VOCAB_SIZE = len(TOKENIZER.vocab)
MODEL_DIM = MODEL.embed_tokens.embedding_dim
PAD_TOKEN = TOKENIZER.pad_token_id
EOS_ID = TOKENIZER.eos_token_id
SOS_ID = TOKENIZER.bos_token_id

# constants

# SOS_ID = 101
# EOS_ID = 102
# MODEL_DIM = 768
# VOCAB_SIZE = 28996

TMP_PATH = Path(paths.tmp_path)
INDEX_FOLDER_PATH = TMP_PATH / paths.index_folder
EMBEDDING_TMP_SUBFOLDER = paths.embedding_tmp_subfolder


def get_tokenizer():
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", "bert-base-cased")
    return TOKENIZER


def get_embedder():
    global MODEL
    if not exists(MODEL):
        MODEL = torch.hub.load("huggingface/pytorch-transformers", "model", "bert-base-cased")
    if torch.cuda.is_available():
        MODEL = MODEL.cuda()

    return MODEL


# tokenize


def tokenize(texts, add_special_tokens=True):
    global TOKENIZER
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    tokenizer = TOKENIZER  # get_tokenizer()

    encoding = tokenizer.batch_encode_plus(
        texts, add_special_tokens=add_special_tokens, padding=True, return_tensors="pt"
    )

    token_ids = encoding.input_ids
    return token_ids


# text to chunks


def doc_text_to_chunks_and_seq_indices(*, doc_text, chunk_size=64, seq_len=2048, pad_id=0):
    assert (seq_len % chunk_size) == 0, "sequence length must be divisible by chunk size"

    ids = tokenize(doc_text)
    ids = rearrange(ids, "1 ... -> ...")

    text_len = ids.shape[-1]

    # pad to multiple of chunk size with an extra token

    padding = chunk_size - ((text_len - 1) % chunk_size)
    ids = F.pad(ids, (0, padding))

    # split out very last token

    ids, last_token = ids[:-1], ids[-1:]
    ids = rearrange(ids, "(n c) -> n c", c=chunk_size)

    # first tokens of chunk [2:] and on will become the last token of chunk [1:]

    last_token_per_chunk = ids[1:, 0]
    all_last_tokens = torch.cat((last_token_per_chunk, last_token), dim=0)
    all_last_tokens = rearrange(all_last_tokens, "n -> n 1")

    # append all last tokens to ids for (num_chunks, chunk_size + 1)

    chunks_with_extra_token = torch.cat((ids, all_last_tokens), dim=-1)

    # calculate chunk indices starting at 0, spaced number of chunks of seq len apart

    total_chunks = ids.shape[0]
    num_chunks_per_seq = seq_len // chunk_size
    seq = torch.arange(0, total_chunks, num_chunks_per_seq)

    return chunks_with_extra_token, seq


def text_folder_to_chunks_(
    *,
    folder,
    chunks_memmap_path,
    seqs_memmap_path,
    doc_ids_memmap_path,
    split_meta_path,
    proj_doc_dict_path,
    doc_proj_dict_path,
    doc_bin_dict_path,
    chunk_size=64,
    seq_len=2048,
    # glob = '**/*.txt',
    data_file_paths,
    max_chunks=1_000_000,
    max_seqs=100_000,  ### ~ total number of sequences in dataset. Sequence has a context size.
):

    total_chunks = 0
    total_docs = 0
    total_seqs = 0

    chunks_shape = (max_chunks, chunk_size + 1)
    seqs_shape = (max_seqs,)
    doc_ids_shape = (max_chunks,)
    file_ind = 0
    split_metainfo = dict()
    proj_doc_dict = defaultdict(list)

    doc_bin_dict = dict()

    for data_file in data_file_paths:
        parquet_file = pq.ParquetFile(data_file)

        column_names = ["bin", "project_id", "doc_id"]
        data = parquet_file.read(column_names).to_pandas()
        doc_bin_dict_split = dict(zip(data["doc_id"], data["bin"]))
        doc_bin_dict.update(doc_bin_dict_split)

    with open(doc_bin_dict_path, "w") as file:
        json.dump(doc_bin_dict, file)

    with memmap(chunks_memmap_path, shape=chunks_shape, dtype=np.int32, mode="w+") as chunks_memmap, memmap(
        seqs_memmap_path, shape=seqs_shape, dtype=np.int32, mode="w+"
    ) as seqs_memmap, memmap(doc_ids_memmap_path, shape=doc_ids_shape, dtype=np.int32, mode="w+") as doc_ids_memmap:
        print("\n ----- Processing code files ------ \n")

        for file in data_file_paths:
            print(f"------ processing {file} -------")
            split = os.path.splitext(os.path.basename(file))[0]
            parquet_file = pq.ParquetFile(file)
            num_row_groups = parquet_file.metadata.num_row_groups

            total_seqs_split = 0

            for i in tqdm(range(num_row_groups)):
                # if i>10:
                #     break
                group = parquet_file.read_row_group(i).to_pandas()

                for index, row in group.iterrows():
                    content = row["content"]
                    doc_id = row["doc_id"]
                    project_id = row["project_id"]
                    proj_doc_dict[project_id].append(doc_id)

                    # chunks - (n_chunks, chunk_len+1) it adds an extra token to the end of each chunk = first token of the next chunk.
                    # seq - [0, num_chunks_per_seq, 2*num_chunks_per_seq, 3*..., ...]
                    chunks, seq = doc_text_to_chunks_and_seq_indices(
                        doc_text=content,
                        chunk_size=chunk_size,
                        seq_len=seq_len,
                    )

                    doc_chunk_len = chunks.shape[0]  # number of chunks in doc
                    doc_seq_len = seq.shape[0]  # number of sequences (512) in doc

                    # adding chunks, seqs and doc_ids
                    # doc_ids - is just a position of the file in a sequence
                    # seqs_memmap - contains chunk ids of the beggining of each sequence.
                    # doc_ids_memmap - doc_id of each chunk

                    chunks_memmap[total_chunks : (total_chunks + doc_chunk_len)] = chunks.numpy()
                    seqs_memmap[total_seqs : (total_seqs + doc_seq_len)] = seq.numpy() + total_chunks
                    doc_ids_memmap[total_chunks : (total_chunks + doc_chunk_len)] = np.full((doc_chunk_len,), doc_id)

                    total_chunks += doc_chunk_len
                    total_seqs += doc_seq_len
                    total_docs += 1

                    total_seqs_split += doc_seq_len

            ## TODO change split_dict. Now dataloader uses split:metadata format.
            split_dict = dict(
                {
                    "split": split,
                    "split size in seqs": total_seqs_split,
                    "first sequence index": total_seqs - total_seqs_split,
                }
            )
            split_metainfo[file_ind] = split_dict
            file_ind += 1

    with open(split_meta_path, "w") as file:
        json.dump(split_metainfo, file)

    for key in proj_doc_dict:
        proj_doc_dict[key].sort()

    doc_proj_dict = {doc_id: project_id for project_id, doc_ids in proj_doc_dict.items() for doc_id in doc_ids}

    with open(doc_proj_dict_path, "w") as file:
        json.dump(doc_proj_dict, file)

    with open(proj_doc_dict_path, "w") as file:
        json.dump(proj_doc_dict, file)

    return dict(chunks=total_chunks, docs=total_docs, seqs=total_seqs, chunk_size=chunk_size)


# embedding function


@torch.no_grad()
def embed(token_ids, return_cls_repr=False, eps=1e-8, pad_id=0, return_all=False):
    global PAD_TOKEN, MODEL
    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    hidden_state = MODEL(token_ids, attention_mask=mask).last_hidden_state

    if return_cls_repr:
        return hidden_state[:, 0]  # return [cls] as representation

    if return_all:
        return hidden_state  # return [cls] as representation

    # if not exists(mask):
    #     return hidden_state.mean(dim=1)

    mask = mask[:, 1:]  # mean all tokens excluding [cls], accounting for length
    mask = rearrange(mask, "b n -> b n 1")

    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)
    return masked_mean


def get_top_similar(retieved, context, k_imp, pad_id=0):

    """
    Takes k_imp tokens from retrieved according to the contexts
    Then pads thiese tokes to match retrieved shape (temporal solution)
    """

    with torch.no_grad():
        context_emb = embed(context, return_all=True)
        retieved_emb = embed(retieved, return_all=True)

    sim = einsum("b i d, b j d -> b i j", context_emb, retieved_emb).sum(dim=1).cpu()
    _, top_ind = torch.topk(sim, k_imp, dim=1)
    important_retrieve = torch.gather(retieved, dim=1, index=top_ind)

    ### padding
    max_seq_len = retieved.size(1)
    pad_lengths = max_seq_len - k_imp
    pad_tokens = torch.full((retieved.size(0), pad_lengths), pad_id, dtype=retieved.dtype, device=retieved.device)
    padded_batch = torch.cat((pad_tokens, important_retrieve), dim=1)

    return padded_batch


# chunks to knn


def chunks_to_embeddings_(
    *,
    num_chunks,
    chunks_memmap_path,
    embeddings_memmap_path,
    chunk_size=64,
    embed_dim=MODEL_DIM,
    batch_size=256,
    use_cls_repr=False,
    pad_id=0.0,
):

    global SOS_ID

    chunks_shape = (num_chunks, chunk_size + 1)
    embed_shape = (num_chunks, embed_dim)

    print(f"Use cls = {use_cls_repr}")

    print("\n -----Embedding ------ \n")

    with memmap(chunks_memmap_path, shape=chunks_shape, dtype=np.int32) as chunks, memmap(
        embeddings_memmap_path, shape=embed_shape, dtype=np.float32, mode="w+"
    ) as embeddings:
        for dim_slice in tqdm(
            range_chunked(num_chunks, batch_size=batch_size),
            total=num_chunks // batch_size,
        ):
            batch_chunk_npy = chunks[dim_slice]
            batch_chunk = torch.from_numpy(batch_chunk_npy)

            cls_tokens = torch.full((batch_chunk.shape[0], 1), SOS_ID)
            batch_chunk = torch.cat((cls_tokens, batch_chunk), dim=1)

            # omit last token, the first token of the next chunk, used for autoregressive training
            batch_chunk = batch_chunk[:, :-1]
            batch_embed = embed(batch_chunk, return_cls_repr=use_cls_repr)
            embeddings[dim_slice] = batch_embed.detach().cpu().numpy()


def memmap_file_to_chunks_(memmap_path, *, folder, shape, dtype, max_rows_per_file=500):
    rows, _ = shape

    with memmap(memmap_path, shape=shape, dtype=dtype, mode="r") as f:
        root_path = TMP_PATH / folder
        reset_folder_(root_path)

        print(f"\n ----- saving to {str(root_path)} ----- \n")
        for ind, dim_slice in tqdm(enumerate(range_chunked(rows, batch_size=max_rows_per_file))):
            filename = root_path / f"{ind:05d}.npy"
            data_slice = f[dim_slice]

            np.save(str(filename), f[dim_slice])
    print("\n ----- saving FINISHED ----- \n")


def build_compound_index(
    data, index_file: str, index_params: dict[Any, Any], d: int, verbose: bool = True, save_to_file: bool = True
) -> Any:
    # m - number of NN edges in a graph HNSW
    # d - vectors dimension
    # efCons - efConstruction controls the size of the dynamic list for the nearest neighbors during the construction of the HNSW index.
    # Increasing this value improves the recall of the index at the cost of longer indexing time.
    # In other words, it's a speed/accuracy trade-off during the index building phase.
    # efSearch - number of inspecting NNs during search time

    index = faiss.IndexHNSWFlat(d, index_params.m)
    index.hnsw.efConstruction = index_params.efCons
    index.hnsw.efSearch = index_params.efSearch

    if verbose:
        index.verbose = index_params.verbose
    else:
        index.verbose = False

    if verbose:
        print("Training index")
    index.train(data)
    if verbose:
        print("Adding data")
    assert index.is_trained
    index.add(data)
    if save_to_file:
        faiss.write_index(index, index_file)

    return index


def calculate_per_project_knn(
    doc_ids_memmap_path,
    embedding_path,
    index_params,
    knn_path,
    proj_doc_dict_path,
    num_chunks,
    num_nearest_neighbors=2,
    num_extra_neighbours=100,
):

    embed_shape = (num_chunks, MODEL_DIM)
    doc_ids = np.memmap(doc_ids_memmap_path, shape=(num_chunks,), dtype=np.int32, mode="r")
    doc_ids = np.array(doc_ids)
    embeddings_all = np.memmap(embedding_path, shape=embed_shape, dtype=np.float32, mode="r")

    with open(proj_doc_dict_path, "r") as file:
        proj_doc_dict = json.load(file)

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

    print(f"KNNs are saved into {knn_path}")
    print(f"Time used = {(time.time() - tt):.2f}")

    error_rates = np.array(error_rates)[:, 1].astype(float)
    error_rates_av = np.mean(error_rates[error_rates > 0])

    print(f"Number of nonzero error rates  = {np.sum(error_rates>0)}")
    print(f"Average nonzero error rate  = {error_rates_av}")
    print(f"Max error rate  = {np.max(error_rates)}")


def test_knn(
    embedding_path,
    knn_path,
    num_chunks,
    num_nearest_neighbors=2,
    n_samples=50_000,
):

    embed_shape = (num_chunks, MODEL_DIM)
    embeddings_all = np.memmap(embedding_path, shape=embed_shape, dtype=np.float32, mode="r")

    knn_map = np.memmap(knn_path, shape=(num_chunks, num_nearest_neighbors), dtype=np.int32, mode="r")
    knn_map = np.array(knn_map)

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


def index_embeddings(
    embedding_path,
    *,
    index_folder,
    index_file="knn.index",
    num_chunks,
    index_params,
):

    global MODEL_DIM
    # embeddings_path = TMP_PATH / embeddings_folder
    index_path = index_folder / index_file

    reset_folder_(INDEX_FOLDER_PATH)

    embeddings = np.memmap(str(embedding_path), dtype=np.float32, mode="r", shape=(num_chunks, MODEL_DIM))

    index = build_compound_index(embeddings, str(index_path), index_params, d=MODEL_DIM)

    return index


def chunks_to_index_and_embed(
    *,
    num_chunks,
    chunk_size,
    chunk_memmap_path,
    index_params,
    use_cls_repr=False,
    max_rows_per_file=500,
    chunks_to_embeddings_batch_size=256,
    embed_dim=MODEL_DIM,
    index_folder,
    index_file="knn.index",
):
    embedding_path = f"{chunk_memmap_path}.embedded"
    embed_shape = (num_chunks, embed_dim)

    if Path(embedding_path).exists():
        print(
            f"----- \n Embeddings file exist: {embedding_path} \n proceeding without creating new embeddings \n ------"
        )
    else:
        chunks_to_embeddings_(
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            chunks_memmap_path=chunk_memmap_path,
            embeddings_memmap_path=embedding_path,
            use_cls_repr=use_cls_repr,
            batch_size=chunks_to_embeddings_batch_size,
            embed_dim=embed_dim,
        )

    index_path = index_folder / index_file
    if index_path.exists():
        print("Found index file. Reading")
        index = faiss_read_index(index_path)
    else:

        # index = index_embeddings(
        #     # embeddings_folder=EMBEDDING_TMP_SUBFOLDER,
        #     embedding_path=embedding_path,
        #     index_folder=index_folder,
        #     index_file=index_file,
        #     num_chunks=num_chunks,
        #     index_params=index_params,
        #     # **index_kwargs,
        # )
        index = None

    embeddings = np.memmap(embedding_path, shape=embed_shape, dtype=np.float32, mode="r")
    return index, embeddings


def chunks_to_precalculated_knn_(
    *,
    num_nearest_neighbors,
    num_chunks,
    chunk_size,
    chunk_memmap_path,
    doc_ids_memmap_path,
    index_params,
    use_cls_repr=False,
    max_rows_per_file=1000,
    chunks_to_embeddings_batch_size=256,
    embed_dim=MODEL_DIM,
    num_extra_neighbors=10,
    force_reprocess=False,
    index_file="knn.index",
    precalculate_knn=False,
):
    chunk_path = Path(chunk_memmap_path)
    knn_path = chunk_path.parents[0] / f"{chunk_path.stem}.knn{chunk_path.suffix}"
    index_path = chunk_path.parents[0] / index_file

    # early return knn path and faiss index
    # unless if force_reprocess is True

    if index_path.exists() and knn_path.exists() and not force_reprocess:
        print(f"Found index file and {chunk_path.stem}.knn{chunk_path.suffix}. Loading.")
        index = faiss_read_index(index_path)
        return knn_path, index

    # fetch the faiss index and calculated embeddings for the chunks

    index, embeddings = chunks_to_index_and_embed(
        num_chunks=num_chunks,
        chunk_size=chunk_size,
        chunk_memmap_path=chunk_memmap_path,
        index_folder=chunk_path.parents[0],
        index_file=index_file,
        index_params=index_params,
        use_cls_repr=use_cls_repr,
        chunks_to_embeddings_batch_size=chunks_to_embeddings_batch_size,
        max_rows_per_file=max_rows_per_file,
    )

    total_neighbors_to_fetch = num_extra_neighbors + num_nearest_neighbors + 1

    print("\n---- Calculating KNNs -----\n")

    with memmap(knn_path, shape=(num_chunks, num_nearest_neighbors), dtype=np.int32, mode="w+") as knns, memmap(
        doc_ids_memmap_path, shape=(num_chunks,), dtype=np.int32, mode="r"
    ) as doc_ids:
        for dim_slice in tqdm(
            range_chunked(num_chunks, batch_size=max_rows_per_file),
            total=num_chunks // max_rows_per_file,
        ):
            query_vector = embeddings[dim_slice]

            distances, indices = index.search(query_vector, k=total_neighbors_to_fetch)

            # remove self from distances and indices
            distances = distances[:, 1:]
            indices = indices[:, 1:]

            # mask out any neighbors that belong to the same document to -1
            query_doc_ids = doc_ids[dim_slice]
            neighbor_doc_ids = doc_ids[indices]
            neighbor_from_same_doc = query_doc_ids[..., None] == neighbor_doc_ids

            indices = np.where(neighbor_from_same_doc, -1, indices)
            distances = np.where(neighbor_from_same_doc, 1e3, distances)

            # re-sort indices by updated distances
            indices = np.take_along_axis(indices, np.argsort(distances, axis=1), axis=1)

            # store nearest neighbors to knn memmap
            knns[dim_slice] = indices[:, :num_nearest_neighbors]

    print(f"knn saved to {knn_path}")
    return knn_path, index
