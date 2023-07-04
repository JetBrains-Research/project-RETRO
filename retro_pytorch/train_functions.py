from typing import Any, Callable, Iterator, TextIO

import torch
from tqdm import tqdm

from retro_pytorch.dataloaders import DataLoaderFromFile, DatasetJsonl


def calc_loss(
    seq: torch.Tensor,
    docs: torch.Tensor,
    model,
    no_retrieve: bool,
    fetch_neighbours: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Any:

    if not no_retrieve:
        retrieved = fetch_neighbours(seq.cuda(), doc_except=docs)
    else:
        retrieved = None

    loss = model(seq.cuda(), retrieved=retrieved, return_loss=True)
    del seq, retrieved
    loss.backward()

    return loss


def grad_step(optimizer: Any, scheduler: Any, loss: Any, loss_train_list: list[float], out_file) -> None:
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    loss_train_list.append(loss.item())
    out_file.write(str(loss.item()) + "\n")
    del loss


def save_model(prefix: str, model: Any, model_folder: str, model_name: str) -> None:
    print(f"---- Saving the {prefix} model -----")
    model_file_name = model_folder + f"{model_name}_{prefix}.pth"
    torch.save(model.state_dict(), model_file_name)


def val_steps(
    model: Any,
    no_retrieve: bool,
    fetch_neighbours: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_val: int,
    val_dl_iter: Iterator,
) -> tuple[list[float], int]:
    model.eval()
    print("------ Validation ------")
    losses_val_cur = []
    for val_step, (seq, docs) in enumerate(tqdm(val_dl_iter, total=num_val, ncols=80), start=1):

        loss = calc_loss(seq, docs, model, no_retrieve, fetch_neighbours)
        losses_val_cur.append(loss.item())
        if num_val is not None:
            if val_step >= num_val:
                break

    return losses_val_cur, val_step


def val_upadate(
    model: Any,
    losses_val: list[float],
    losses_val_cur: list[float],
    model_folder: str,
    model_name: str,
    val_dl_iter: Iterator,
    f_val: TextIO,
    max_val_loss: float,
    saved_ind: int,
) -> tuple[float, int, Iterator]:

    if len(losses_val_cur) != 0:
        loss_cur = sum(losses_val_cur) / (len(losses_val_cur))
        losses_val.append(loss_cur)
        f_val.write(str(loss_cur) + "\n")
        f_val.flush()

        save_model("last", model, model_folder, model_name)

        if loss_cur < max_val_loss:
            max_val_loss = loss_cur
            save_model(f"best_{saved_ind}", model, model_folder, model_name)
            saved_ind = (saved_ind + 1) % 3

    return max_val_loss, saved_ind, val_dl_iter
