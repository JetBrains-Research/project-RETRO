from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from retro_pytorch.retrieval import EOS_ID
from retro_pytorch.utils import memmap

# knn to retrieved chunks


def knn_to_retrieved_chunks(
    knns,
    chunks_memmap,
    *,
    add_continuations,
    num_chunks,
    pad_id=0,
    eos_id=EOS_ID,
):
    # derive mask for no neighbors found (-1)

    no_neighbor_mask = knns == -1
    knns = np.maximum(knns, 0)

    # get neighbor and continuation chunks

    knn_chunks = chunks_memmap[knns]
    is_last_document_chunk = np.any(knn_chunks == eos_id, axis=-1, keepdims=True)

    # use presence of [EOS] in chunk as way to detect document boundaries
    # [EOS] in BERT tokenizer is 102

    retrieved = knn_chunks[..., :-1]

    if add_continuations:
        continuation_indices = np.clip(knns + 1, 0, num_chunks - 1)  # chunks are stored contiguously
        continuation_chunks = chunks_memmap[continuation_indices][..., :-1]
        continuation_chunks *= ~is_last_document_chunk

        # combine neighbors with continuations

        retrieved = np.concatenate((retrieved, continuation_chunks), axis=-1)

    # mask out any nearest neighbor chunks that was -1 (not found at index time) to padding id

    retrieved = np.where(~no_neighbor_mask[..., None], retrieved, pad_id)
    return retrieved


# dataset


class RETRODataset(Dataset):
    def __init__(
        self,
        *,
        num_chunks,
        chunk_size,
        seq_len,
        num_sequences,
        num_neighbors,
        chunk_memmap_path,
        chunk_nn_memmap_path,
        seq_memmap_path,
        eos_id=EOS_ID,
        pad_id=0.0,
        add_continuations=True,
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.num_sequences = num_sequences
        self.seq_num_chunks = seq_len // chunk_size
        self.eos_id = eos_id
        self.pad_id = pad_id

        num_chunks_with_padding = num_chunks + self.seq_num_chunks

        chunks_shape = (num_chunks_with_padding, chunk_size + 1)
        knn_shape = (num_chunks_with_padding, num_neighbors)

        self.add_continuations = add_continuations
        self.get_chunks = partial(memmap, chunk_memmap_path, dtype=np.int32, shape=chunks_shape)
        self.get_knns = partial(memmap, chunk_nn_memmap_path, dtype=np.int32, shape=knn_shape)
        self.get_seqs = partial(memmap, seq_memmap_path, dtype=np.int32, shape=(num_sequences,))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, ind):
        # global begin_chunk_index, chunk_range, chunks, seq_tokens_, seq_tokens, seq_mask, seq_mask_
        with self.get_chunks() as chunks_memmap, self.get_knns() as knns_memmap, self.get_seqs() as seqs_memmap:
            begin_chunk_index = seqs_memmap[ind]
            chunk_range = slice(begin_chunk_index, (begin_chunk_index + self.seq_num_chunks))
            # print(chunk_range)
            chunks = chunks_memmap[chunk_range]

            # excise the last token, except for last token of last chunk

            seq_tokens = np.concatenate((chunks[:, :-1].flatten(), chunks[-1, -1:]))

            # mask out (with padding tokens) any token following an <eos> | disallow having more than 1 document in a sequence, as it would break RETRO's CCA

            seq_mask = np.cumsum(seq_tokens == self.eos_id, axis=0)
            seq_mask = np.pad(seq_mask, (1, 0))[:-1] == 0.0
            seq_tokens = np.where(seq_mask, seq_tokens, 0.0)

            # derive retrieved tokens
            knns = knns_memmap[chunk_range]

            retrieved = knn_to_retrieved_chunks(
                knns,
                chunks_memmap,
                add_continuations=self.add_continuations,
                eos_id=self.eos_id,
                num_chunks=self.num_chunks,
            )

        seq_tokens_torch = torch.from_numpy(seq_tokens).long()
        retrieved_torch = torch.from_numpy(retrieved).long()
        return seq_tokens_torch, retrieved_torch


"""
Dataset for the reading data from jsonl file
"""

import os

import jsonlines
from torch.utils.data import DataLoader

from retro_pytorch.retrieval import doc_text_to_chunks_and_seq_indices


def split_into_chunks(seq_tokens, seq_length, pad_id=0):
    # Calculate the number of chunks needed
    num_chunks = len(seq_tokens) // seq_length
    remainder = len(seq_tokens) % seq_length

    # Split the array into chunks
    chunks = np.split(seq_tokens[: num_chunks * seq_length], num_chunks)

    # Pad the last chunk if needed
    if remainder > 0:
        last_chunk = np.pad(
            seq_tokens[num_chunks * seq_length :],
            (pad_id, seq_length - remainder),
            mode="constant",
        )
        chunks.append(last_chunk)

    return chunks


class Dataset_jsonl(Dataset):
    def __init__(self, file_path, cnunk_size=64, seq_length=512, pad_id=0):
        self.file_path = file_path
        self.chunk_size = cnunk_size
        self.seq_length = seq_length
        self.chunks_in_seq = seq_length // cnunk_size
        file_size = os.path.getsize(self.file_path)
        self.length = file_size // 200
        self.pad_id = pad_id

    def __iter__(self):
        ### returns sequences of concatinated 8 chuncks + last token (8*64 + 1) = 513
        with jsonlines.open(self.file_path) as reader:
            for line in reader:
                try:
                    content = line["content"]
                except KeyError:
                    content = line["contents"]
                doc_id = self.chunks_in_seq * [line["doc_id"]]
                chunks, seq = doc_text_to_chunks_and_seq_indices(
                    doc_text=content,
                    chunk_size=self.chunk_size,
                    seq_len=self.seq_length,
                )

                seq_chunks = torch.split(chunks, self.chunks_in_seq, dim=0)

                for seq in seq_chunks:
                    seq_tokens = torch.concat((seq[:, :-1].flatten(), seq[-1, -1:]))
                    if len(seq_tokens) < self.seq_length + 1:
                        seq_tokens = F.pad(
                            seq_tokens,
                            (self.pad_id, self.seq_length + 1 - len(seq_tokens)),
                        )
                    yield seq_tokens, doc_id

    def __getitem__(self, index):
        raise NotImplementedError("We want to use __iter__ instead")

    def __len__(self):
        # Return an estimate or approximation of the total number of examples
        # Alternatively, you can return a large number or None to indicate an unknown length
        return self.length


class DataLoader_from_file(DataLoader):
    def __init__(self, dataset, batch_size=1, collate_fn=None):
        self.collate_fn = collate_fn
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # Create an iterator object for your custom dataset
        iterator = iter(self.dataset)

        while True:
            # Accumulate items from the iterator
            batch_items = []
            batch_docs = []
            for _ in range(self.batch_size):
                try:
                    item, doc_id = next(iterator)
                    batch_items.append(item)
                    batch_docs.append(doc_id)
                except StopIteration:
                    break

            if len(batch_items) == 0:
                break

            # Apply collate_fn to the batch items, if one is provided
            if self.collate_fn is not None:
                batch_items = self.collate_fn(batch_items)

            batch_items = torch.stack(batch_items)
            batch_docs = torch.tensor(batch_docs)

            yield batch_items, batch_docs
