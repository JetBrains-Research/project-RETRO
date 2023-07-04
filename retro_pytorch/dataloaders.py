import os
from typing import Callable, Iterator, Optional, Union

import jsonlines
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from retro_pytorch.retrieval import doc_text_to_chunks_and_seq_indices

"""
Dataset for the reading data from jsonl file
"""


class DatasetJsonl(Dataset):
    def __init__(self, file_path: str, cnunk_size: int = 64, seq_length: int = 512, pad_id: int = 0):
        self.file_path = file_path
        self.chunk_size = cnunk_size
        self.seq_length = seq_length
        self.chunks_in_seq = seq_length // cnunk_size
        file_size = os.path.getsize(self.file_path)
        self.length = file_size // 200
        self.pad_id = pad_id

    def __iter__(self) -> Iterator[tuple[torch.Tensor, list[int]]]:
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

    def __getitem__(self, index: int) -> None:
        raise NotImplementedError("We want to use __iter__ instead")

    def __len__(self) -> int:
        # Return an estimate or approximation of the total number of examples
        # Alternatively, you can return a large number or None to indicate an unknown length
        return self.length


class DataLoaderFromFile(DataLoader):
    def __init__(
        self,
        dataset: "DatasetJsonl",
        batch_size: int = 1,
        collate_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.collate_fn = collate_fn
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
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

            # batch_items = torch.stack(batch_items)
            # batch_docs = torch.tensor(batch_docs)

            yield torch.stack(batch_items), torch.tensor(batch_docs)
