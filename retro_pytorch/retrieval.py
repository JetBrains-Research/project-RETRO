from pathlib import Path
from typing import Any

import faiss
import jsonlines
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
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
    chunk_size=64,
    seq_len=2048,
    # glob = '**/*.txt',
    data_file_paths,
    max_chunks=1_000_000,
    max_seqs=100_000,  ### ~ total number of sequences in dataset. Sequence has a context size.
):
    # paths = sorted([*Path(folder).glob(glob)])

    total_chunks = 0
    total_docs = 0
    total_seqs = 0

    chunks_shape = (max_chunks, chunk_size + 1)
    seqs_shape = (max_seqs,)
    doc_ids_shape = (max_chunks,)

    with memmap(chunks_memmap_path, shape=chunks_shape, dtype=np.int32, mode="w+") as chunks_memmap, memmap(
        seqs_memmap_path, shape=seqs_shape, dtype=np.int32, mode="w+"
    ) as seqs_memmap, memmap(doc_ids_memmap_path, shape=doc_ids_shape, dtype=np.int32, mode="w+") as doc_ids_memmap:
        print("\n ----- Processing code files ------ \n")
        for file in data_file_paths:
            print(f"------ processing {file} -------")
            reader = jsonlines.open(file)

            for line in tqdm(reader, total=330_000, ncols=100):
                content = line["contents"]
                doc_id = line["doc_id"]

                # chunks - (n_chunks, chunk_len+1) it adds an extra token to the end of each chunk = first token of the next chunk.
                # seq - [0, num_chunks_per_seq, 2*num_chunks_per_seq, 3*..., ...]
                chunks, seq = doc_text_to_chunks_and_seq_indices(
                    doc_text=content,
                    chunk_size=chunk_size,
                    seq_len=seq_len,
                )

                doc_chunk_len = chunks.shape[0]  # number of chunks in doc
                doc_seq_len = seq.shape[0] # number of sequences (512) in doc

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
    

    return dict(chunks=total_chunks, docs=total_docs, seqs=total_seqs, chunk_size=chunk_size)


# embedding function


@torch.no_grad()
def embed(token_ids, return_cls_repr=False, eps=1e-8, pad_id=0):
    global PAD_TOKEN, MODEL
    model = MODEL  # get_embedder()
    model.eval()
    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    hidden_state = model(token_ids, attention_mask=mask).last_hidden_state

    if return_cls_repr:
        return hidden_state[:, 0]  # return [cls] as representation

    # if not exists(mask):
    #     return hidden_state.mean(dim=1)

    mask = mask[:, 1:]  # mean all tokens excluding [cls], accounting for length
    mask = rearrange(mask, "b n -> b n 1")

    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)
    return masked_mean


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


def build_compound_index(data, index_file: str, index_params: dict[Any, Any], d: int, verbose: bool = True, save_to_file: bool = True) -> Any:
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
        index = index_embeddings(
            # embeddings_folder=EMBEDDING_TMP_SUBFOLDER,
            embedding_path=embedding_path,
            index_folder=index_folder,
            index_file=index_file,
            num_chunks=num_chunks,
            index_params=index_params,
            # **index_kwargs,
        )

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
    max_rows_per_file=500,
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
    )

    """
    I switched off the precalculation of KNNs as I do not use it for now.
    """

    if precalculate_knn:
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

                # print(f'knns calculated for {dim_slice.stop} / {num_chunks}')

        print(f"knn saved to {knn_path}")
    return knn_path, index
