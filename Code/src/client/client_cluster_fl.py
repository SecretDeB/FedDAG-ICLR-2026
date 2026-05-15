import torch.utils.data as data

import numpy as np
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F
import random
from torch.utils.data import random_split
from collections import OrderedDict


class Client_ClusterFL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_dl_local=None, test_dl_local=None,
                 pre_train_dl=None, validation_dl=None):

        self.name = name
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()

        # Use AMP automatically if running on CUDA
        if isinstance(self.device, torch.device):
            self.use_amp = (self.device.type == "cuda")
        else:
            # in case someone passes a string like "cuda:0"
            self.use_amp = "cuda" in str(self.device)

        # GradScaler for primary training (train / train2) – new API
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0
        self.count = 0
        self.save_best = True
        self.pre_train_dl = pre_train_dl
        self.validation_dl = validation_dl

    # --------------------------------------------------
    # Primary local training (full model)
    # --------------------------------------------------
    def train(self, is_print: bool = False):
        self.count += 1
        self.net.to(self.device)
        self.net.train()

        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=0,
        )

        epoch_loss = []
        for _ in range(self.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    # NEW AMP API
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        logits = self.net(images)
                        loss = self.loss_func(logits, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logits = self.net(images)
                    loss = self.loss_func(logits, labels)
                    loss.backward()
                    optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(np.mean(batch_loss))

        return float(np.mean(epoch_loss))

    # --------------------------------------------------
    # Primary local training (only trainable params)
    # --------------------------------------------------
    def train2(self, is_print: bool = False):
        self.count += 1
        self.net.to(self.device)
        self.net.train()

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.net.parameters()),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=0,
        )

        epoch_loss = []
        for _ in range(self.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    # NEW AMP API
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        logits = self.net(images)
                        loss = self.loss_func(logits, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logits = self.net(images)
                    loss = self.loss_func(logits, labels)
                    loss.backward()
                    optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(np.mean(batch_loss))

        return float(np.mean(epoch_loss))

    # ------------------------------------------------------------
    # SAFE secondary-encoder enrichment (leaves self.net unchanged)
    # ------------------------------------------------------------
    def train_secondary(self,
                        base_sec_state: dict,
                        epochs: int = 1,
                        lr: float = 0.01) -> dict:
        """
        Fine-tune a *copy* of base_sec_state on local data and
        return the **parameter delta** Δθ = θ_i' − base_sec_state.

        Result is returned on CPU so server-side aggregation
        (w_glob_per_cluster[j][key] += dv) works without device issues.
        """
        # 1. clone full model and move to device
        temp = copy.deepcopy(self.net).to(self.device)

        # 2. load base weights into secondary encoder only
        temp.secondary_encoder.load_state_dict(base_sec_state)

        # 3. freeze / unfreeze parts
        for p in temp.own_encoder.parameters():
            p.requires_grad = False
        for p in temp.classifier.parameters():
            p.requires_grad = False
        for p in temp.secondary_encoder.parameters():
            p.requires_grad = True

        # 4. optimizer for secondary encoder
        opt = torch.optim.SGD(
            temp.secondary_encoder.parameters(),
            lr=lr,
            momentum=self.momentum,
        )

        # local scaler for this optimizer (separate from main scaler) – new API
        use_amp = self.use_amp
        sec_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # 5. local training (single loop, AMP-aware)
        temp.train()
        for _ in range(epochs):
            for x, y in self.ldr_train:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                if use_amp:
                    # NEW AMP API
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = temp(x)
                        loss = self.loss_func(out, y)
                    sec_scaler.scale(loss).backward()
                    sec_scaler.step(opt)
                    sec_scaler.update()
                else:
                    out = temp(x)
                    loss = self.loss_func(out, y)
                    loss.backward()
                    opt.step()

        # 6. compute Δθ on CPU so aggregation is CPU-safe
        delta = OrderedDict()
        temp_state = temp.secondary_encoder.state_dict()

        for k, v in temp_state.items():
            v_cpu = v.detach().cpu()
            base_cpu = base_sec_state[k].detach().cpu()
            delta[k] = v_cpu - base_cpu

        return delta

    # --------------------------------------------------
    # Pre-training (no AMP yet, can be extended)
    # --------------------------------------------------
    def pre_train(self, is_print=False):
        self.net.to(self.device)
        self.net.train()

        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=0,
        )

        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.pre_train_dl):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    # --------------------------------------------------
    # Utility getters / setters
    # --------------------------------------------------
    def get_state_dict(self):
        return self.net.state_dict()

    def get_best_acc(self):
        return self.acc_best

    def get_count(self):
        return self.count

    def get_net(self):
        return self.net

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def get_W(self):
        W = {key: copy.deepcopy(value) for key, value in self.net.named_parameters()}
        return W

    # --------------------------------------------------
    # Evaluation helpers
    # --------------------------------------------------
    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_test2(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        l = 0
        with torch.no_grad():
            for data, target in self.validation_dl:
                l += len(data)
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= l
        accuracy = 100. * correct / l
        return test_loss, accuracy

    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy

    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy
