from typing import Any, Generator

import pytorch_lightning as pl
import torch
from torch.optim import AdamW

# from torch.optim.lr_scheduler import StepLR
from transformers import get_cosine_schedule_with_warmup


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model,
        train_parameters,
        no_retrieve=False,
        n_prepend=1,
        self_retr_fun=None,
        retrieve_functions=None,
        fun_names=None,
    ):
        super().__init__()
        self.model = model
        self.train_parameters = train_parameters
        self.val_metric_names = ["loss", "recall@1", "recall@3", "recall@5"]
        # self.save_hyperparameters()
        self.retrieve_functions = retrieve_functions
        self.self_retr_fun = self_retr_fun
        self.fun_names = fun_names
        self.no_retrieve = no_retrieve
        self.n_prepend = n_prepend

    def training_step(self, batch, batch_idx):
        seq, ret = batch
        if not self.no_retrieve:
            ret = self.self_retr_fun(seq, ret=ret, n_prepend=self.n_prepend)
            ret = ret[self.n_prepend :]
        else:
            if self.n_prepend == 2:
                ret1 = seq[:-2, :-1]
                ret2 = seq[1:-1, :-1]
                ret = torch.cat((ret1, ret2), dim=-1)
            elif self.n_prepend == 1:
                ret = seq[:-1, :-1]

        seq = seq[self.n_prepend :]
        loss = self.model(seq, retrieved=ret, return_loss=True)
        self.log_dict({"train/loss": loss.item()}, on_step=True, prog_bar=True, logger=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr)

    def configure_optimizers(self):

        lr = self.train_parameters["lr"]
        wd = self.train_parameters["wd"]
        warmup_steps = self.train_parameters["warmup_steps"]
        training_steps = self.train_parameters["training_steps"]

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def validation_step(self, batch, batch_idx):

        seq, ret = batch
        losses = []
        for fetch_fn in self.retrieve_functions:
            retrieved = fetch_fn(seq, ret=ret, n_prepend=self.n_prepend)
            # cut the batch size, so that retrieved and continuation seqs would come in batch of same size
            if seq.size(0) == retrieved.size(0):
                retrieved = retrieved[self.n_prepend :]
            seq_cut = seq[self.n_prepend :]

            val_loss = self.model(seq_cut, retrieved=retrieved, return_loss=True, return_recall=True, k_list=[1, 3, 5])
            losses.append(val_loss)
        losses = torch.stack(losses, dim=1)
        metric_dict = dict()
        for fun_name, i in zip(self.fun_names, range(losses.size(-1))):
            losses_fun = losses[:, i]
            metric_dict.update(
                {"val/" + name + fun_name: val.cpu() for name, val in zip(self.val_metric_names, losses_fun)}
            )
        self.log_dict(metric_dict)

    def test_step(self, batch, batch_idx):
        pass
