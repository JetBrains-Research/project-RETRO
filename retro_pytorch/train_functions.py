from typing import Any, Callable, Iterator, TextIO

import numpy as np
import torch
from tqdm import tqdm

# def calc_loss(
#     seq: torch.Tensor,
#     docs: torch.Tensor,
#     model,
#     no_retrieve: bool,
#     fetch_neighbours: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
# ) -> Any:

#     if not no_retrieve:
#         retrieved = fetch_neighbours(seq.cuda(), doc_except=docs)
#     else:
#         retrieved = None

#     loss = model(seq.cuda(), retrieved=retrieved, return_loss=True)
#     if loss.requires_grad:
#         loss.backward()

#     return loss

# def calc_loss_concat(
#     seq: torch.Tensor,
#     docs: torch.Tensor,
#     model,
#     fetch_neighbours: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
# ) -> Any:
#     retrieved = fetch_neighbours(seq.cuda(), doc_except=docs)

#     retrieve = torch.reshape(retrieved[:, :, 0, 64:], (retrieved.size(0), retrieved.size(1) * retrieved.size(-1) // 2))
#     seq_concat = torch.concat((retrieve, seq.cuda()), dim=-1)

#     loss = model(seq_concat.cuda(), retrieved=None, return_loss=True, seq_len=512)
#     if loss.requires_grad:
#         loss.backward()

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
    val_dl_iter: Iterator,
    num_val: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    batch_aggregate = []
    val_step = 0
    for val_step, (seq, docs) in enumerate(val_dl_iter, start=1):
        batch_aggregate.append((seq, docs))
        if num_val is not None:
            if val_step >= num_val:
                break

    return batch_aggregate, val_step


def val_steps(
    model: Any,
    val_dl,
    num_val: int,
    no_retrieve: bool,
    fetch_neighbours_list: list[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> list[float]:
    model.eval()
    losses_val_cur = []
    val_step = 0
    with torch.no_grad():
        for seq, ret1, ret2 in tqdm(val_dl, total=num_val, ncols=50):
            seq = seq.cuda()
            if no_retrieve:
                loss = model(seq, retrieved=None, return_loss=True)
                losses_val_cur.append([loss.item()])
            else:

                loss1 = model(seq, retrieved=ret1.cuda(), return_loss=True)
                loss2 = model(seq, retrieved=ret2.cuda(), return_loss=True)
                loss_none = model(seq, retrieved=None, return_loss=True)

                losses = [loss1.item(), loss2.item(), loss_none.item()]

                for fetch_neighbours in fetch_neighbours_list:
                    retrieved = fetch_neighbours(seq)
                    loss = model(seq, retrieved=retrieved.cuda(), return_loss=True)
                    losses.append(loss.item())

                losses_val_cur.append(losses)

            val_step += 1
            if val_step >= num_val:
                break

    return np.array(losses_val_cur), val_step


def val_update(
    model: Any,
    losses_val: list[list[float]],
    losses_val_cur: list[list[float]],
    model_folder: str,
    model_name: str,
    f_val: TextIO,
    max_val_loss: float,
    saved_ind: int,
    saved_last_ind: int,
) -> tuple[float, int, int, Iterator]:

    if len(losses_val_cur) != 0:
        loss_cur = np.mean(losses_val_cur, axis=0).tolist()
        losses_val.append(loss_cur)
        f_val.write(str(loss_cur) + "\n")
        f_val.flush()

        save_model("last_" + str(saved_last_ind), model, model_folder, model_name)
        saved_last_ind = (saved_last_ind + 1) % 3

        # if loss_cur[0] < max_val_loss:
        #     max_val_loss = loss_cur[0]
        #     save_model(f"best_{saved_ind}", model, model_folder, model_name)
        #     saved_ind = (saved_ind + 1) % 3

    return max_val_loss, saved_ind, saved_last_ind
