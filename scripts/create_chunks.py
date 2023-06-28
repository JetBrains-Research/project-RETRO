import torch
import sys
from exit
retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.training import TrainingWrapper
import gc
import time

'''
Create your chunks and chunk start indices (for calculating sequence ranges for autoregressive training) using text_folder_to_chunks_
Creates embeddings and finding knns for each chuncks in dataset
'''

#for n_chuncks, split in zip([2_000_000, 2_000_000, 20_000_000], ['val', 'test', 'train']):
#for n_chuncks, split in zip([20_000_000], ['train']):

n_chuncks = 15_000_000
#n_chuncks = 1_000_000

#texts_folder = '../../data/texts_folder/'
texts_folder = '../../data/texts_folder/'
data_folder = '../../data/full_dataset/'
#print('-------' + split + '-------')

gc.collect()
torch.cuda.empty_cache()

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(
    max_seq_len = 512,                      # max sequence length
    enc_dim = 768,                           # encoder model dimension
    enc_depth = 3,                           # encoder depth
    dec_dim = 768,                           # decoder model dimensions
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
).cuda()

tt = time.time()

wrapper = TrainingWrapper(
    retro = retro,                                 # path to retro instance
    knn = 2,                                       # knn (2 in paper was sufficient)
    chunk_size = 64,                               # chunk size (64 in paper)
    documents_path = data_folder,                  # path to folder of text
    #glob = '**/*.txt',                             # text glob
    data_file_paths = [data_folder + 'val.jsonl', data_folder + 'test.jsonl', data_folder + 'train.jsonl'],
    #data_file_paths = [data_folder + 'val.jsonl'],
    chunks_memmap_path = texts_folder + 'train.chunks.dat',     # path to chunks
    seqs_memmap_path = texts_folder + 'train.seq.dat',          # path to sequence data
    doc_ids_memmap_path = texts_folder + 'train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
    processed_stats_json_path=texts_folder + 'processed-stats.json',
    max_chunks = n_chuncks,                        # maximum cap to chunks
    max_seqs = n_chuncks//5,                            # maximum seqs
    knn_extra_neighbors = 100,                     # num extra neighbors to fetch
    max_index_memory_usage = '40G',
    current_memory_available = '64G'
)

time_used = time.time() - tt
print(f'Time used = {time_used:.2f} s')














