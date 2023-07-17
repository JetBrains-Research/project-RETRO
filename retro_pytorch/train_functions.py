import time
from typing import Any, Callable, Iterator, TextIO

import torch
from tqdm import tqdm

# from transformers import AutoTokenizer
# from omegaconf import OmegaConf
#
# conf_load = OmegaConf.load('config.yaml')
# paths = conf_load.paths
# tokenizer = AutoTokenizer.from_pretrained(paths.encoder_path)
#
# def decode(tens):
#     return tokenizer.batch_decode(tens, skip_special_tokens=True)


def calc_loss(
    seq: torch.Tensor,
    docs: torch.Tensor,
    model,
    no_retrieve: bool,
    fetch_neighbours_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fetch_neighbours_fn_contrast: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    chunk_iter=None,
) -> Any:

    if not no_retrieve:
        retrieved = fetch_neighbours_fn(seq, doc_except=docs, chunk_iter=chunk_iter)
        if fetch_neighbours_fn_contrast is not None:
            retrieved_contrast = fetch_neighbours_fn_contrast(seq, doc_except=docs, chunk_iter=chunk_iter)
        else:
            retrieved_contrast = None
    else:
        retrieved = None

    loss = model(seq.cuda(), retrieved=retrieved, retrieved_contrast=retrieved_contrast, return_loss=True)
    if loss.requires_grad:
        loss.backward()

    return loss


# def calc_loss_concat(
#         seq: torch.Tensor,
#         docs: torch.Tensor,
#         model,
#         fetch_neighbours: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
# ) -> Any:
#     retrieved = fetch_neighbours(seq.cuda(), doc_except=docs)
#
#     seq_concat = torch.reshape(retrieved[:, :, 0, :], (retrieved.size(0), retrieved.size(1) * retrieved.size(-1)))
#     seq_concat = torch.concat((seq_concat, seq.cuda()), dim=-1)
#
#     loss = model(seq_concat.cuda(), retrieved=None, return_loss=True, mask_concat=True)
#     if loss.requires_grad:
#         loss.backward()
#
#     return loss


def grad_step(optimizer: Any, scheduler: Any, loss: Any, loss_train_list: list[float], out_file) -> None:
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    loss_train_list.append(loss.item())
    out_file.write(str(loss.item()) + "\n")


def save_model(prefix: str, model: Any, model_folder: str, model_name: str) -> None:
    print(f"---- Saving the {prefix} model -----")
    model_file_name = model_folder + f"{model_name}_{prefix}.pth"
    torch.save(model.state_dict(), model_file_name)


def aggregate_batches(
    dl_iter: Iterator,
    num_steps: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    batch_aggregate = []
    step = 0
    for step, (seq, docs) in enumerate(dl_iter, start=1):
        batch_aggregate.append((seq, docs))
        if num_steps is not None:
            if step >= num_steps:
                break

    return batch_aggregate, step


def val_steps(
    model: Any,
    no_retrieve: bool,
    fetch_neighbours_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aggregate: list[tuple[torch.Tensor, torch.Tensor]],
    verbose: bool = True,
    chunk_iter=None,
) -> list[float]:
    model.eval()
    losses_val_cur = []
    with torch.no_grad():
        for seq, docs in tqdm(aggregate, ncols=100, disable=not verbose):
            loss = calc_loss(seq, docs, model, no_retrieve, fetch_neighbours_fn, chunk_iter=chunk_iter)
            losses_val_cur.append(loss.item())

    return losses_val_cur


# def val_steps_concat(
#     model: Any,
#     no_retrieve: bool,
#     fetch_neighbours: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
#     aggregate: list[tuple[torch.Tensor, torch.Tensor]],
# ) -> list[float]:
#     model.eval()
#     losses_val_cur = []
#     with torch.no_grad():
#         for seq, docs in tqdm(aggregate, ncols=100):
#             loss = calc_loss_concat(seq, docs, model, fetch_neighbours)
#             losses_val_cur.append(loss.item())
#
#     return losses_val_cur


def val_update(
    model: Any,
    losses_val: list[list[float]],
    losses_val_cur: list[list[float]],
    model_folder: str,
    model_name: str,
    val_dl_iter: Iterator,
    f_val: TextIO,
    max_val_loss: float,
    saved_ind: int,
    saved_last_ind: int,
) -> tuple[float, int, int, Iterator]:

    if len(losses_val_cur) != 0:
        loss_cur = [sum(losses_cur) / (len(losses_cur)) for losses_cur in losses_val_cur]
        losses_val.append(loss_cur)
        f_val.write(str(loss_cur) + "\n")
        f_val.flush()

        save_model("last_" + str(saved_last_ind), model, model_folder, model_name)
        saved_last_ind = (saved_last_ind + 1) % 3

        # if loss_cur[0] < max_val_loss:
        #     max_val_loss = loss_cur[0]
        #     save_model(f"best_{saved_ind}", model, model_folder, model_name)
        #     saved_ind = (saved_ind + 1) % 3

    return max_val_loss, saved_ind, saved_last_ind, val_dl_iter
