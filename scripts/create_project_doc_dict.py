from retro_pytorch.utils import seed_all

seed_all(1111)
import argparse
import json
import jsonlines
from collections import defaultdict
import os
from tqdm import tqdm

from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description="")
parser.add_argument("-no", "--no-retrieve", action="store_true", help="Do not retrieve if flag added")
parser.add_argument("-config", "--config", default="config_dev.yaml", help="Config filename")
args = parser.parse_args()
no_retrieve = args.no_retrieve
config_name = args.config

## loading pathes
print(f"Loading configs from {config_name} file")
config = OmegaConf.load(config_name)
paths = config.paths

# %%

data_file_paths=[
    os.path.join(paths.data_folder, "val.jsonl"),
    os.path.join(paths.data_folder, "test.jsonl"),
    os.path.join(paths.data_folder, "train.jsonl")
    ]

proj_doc_dict = defaultdict(list)

for file in data_file_paths:

    with jsonlines.open(file) as reader:
        for line in tqdm(reader):
            doc_id = line["doc_id"]
            project_id = line["project_id"]
            proj_doc_dict[project_id].append(doc_id)

for key in proj_doc_dict:
    proj_doc_dict[key].sort()

with open(paths.data_folder + "proj_doc_dict.json", "w") as file:
    json.dump(proj_doc_dict, file)
# doc_proj_dict = dict()
# for project_id, doc_ids in proj_doc_dict.items():
#     for doc_id in doc_ids:
#         doc_proj_dict[doc_id] = project_id
        
# proj_size_dict = dict()
# for project_id in proj_doc_dict.keys():
#     proj_size_dict[project_id] = len(proj_doc_dict[project_id])

# %%


# with open(paths.data_folder + "doc_proj_dict.json", "w") as file:
#     json.dump(doc_proj_dict, file)
    
# with open(paths.data_folder + "proj_size_dict.json", "w") as file:
#     json.dump(proj_size_dict, file)

# %%