from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset

from retro_pytorch.retrieval import EOS_ID
from retro_pytorch.utils import memmap

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
        eos_id = EOS_ID,
        pad_id = 0.
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.num_sequences = num_sequences
        self.seq_num_chunks = seq_len // chunk_size
        self.eos_id = eos_id
        self.pad_id = pad_id

        shape = (num_chunks, chunk_size + 1)

        self.get_chunks = partial(memmap, chunk_memmap_path, dtype = np.int32, shape = shape)
        self.get_knns = partial(memmap, chunk_nn_memmap_path, dtype = np.int32, shape = (num_chunks, num_neighbors))
        self.get_seqs = partial(memmap, seq_memmap_path, dtype = np.int32, shape = (num_sequences,))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, ind):
        with self.get_chunks() as chunks_memmap, self.get_knns() as knns_memmap, self.get_seqs() as seqs_memmap:

            begin_chunk_index = seqs_memmap[ind]
            chunk_range = slice(begin_chunk_index, (begin_chunk_index + self.seq_num_chunks))

            chunks = chunks_memmap[chunk_range]

            # excise the last token, except for last token of last chunk

            seq_tokens = np.concatenate((chunks[:, :-1].flatten(), chunks[-1, -1:]))

            # mask out (with padding tokens) any token following an <eos> | disallow having more than 1 document in a sequence, as it would break RETRO's CCA

            seq_mask = np.cumsum(seq_tokens == self.eos_id, axis = 0)
            seq_mask = np.pad(seq_mask, (1, 0))[:-1] == 0.
            seq_tokens = np.where(seq_mask, seq_tokens, 0.)

            # derive retrieved tokens

            knns = knns_memmap[chunk_range]

            # derive mask for no neighbors found (-1)

            no_neighbor_mask = knns == -1
            knns = np.maximum(knns, 0)

            # get neighbor and continuation chunks

            knn_chunks = chunks_memmap[knns]
            is_last_document_chunk = np.any(knn_chunks == self.eos_id, axis = -1, keepdims = True)

            # use presence of [EOS] in chunk as way to detect document boundaries
            # [EOS] in BERT tokenizer is 102

            knn_chunks = knn_chunks[..., :-1]

            continuation_indices = np.clip(knns + 1, 0, self.num_chunks - 1) # chunks are stored contiguously
            continuation_chunks = chunks_memmap[continuation_indices][..., :-1]
            continuation_chunks *= ~is_last_document_chunk

            # combine neighbors with continuations

            retrieved = np.concatenate((knn_chunks, continuation_chunks), axis = -1)

            # mask out any nearest neighbor chunks that was -1 (not found at index time) to padding id

            retrieved = np.where(~no_neighbor_mask[..., None], retrieved, self.pad_id)

        seq_tokens_torch = torch.from_numpy(seq_tokens).long()
        retrieved_torch = torch.from_numpy(retrieved).long()
        return seq_tokens_torch, retrieved_torch
