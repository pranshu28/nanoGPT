import copy
from itertools import product

import numpy as np
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from math import sqrt

from typing import Tuple, Optional, Callable
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from torch import Tensor

import math
from copy import deepcopy

from torch.optim import Optimizer
import torch.nn.functional as F

Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]

LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
OptFloat = Optional[float]


class Adabeta(Optimizer):
    def __init__(self, params, lr=0.001, momentum=0.9, const=1e-8, cosim_decay=0.9,
                 weight_decay=0, use_sgdm=False, nesterov=False):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, const=const, cosim_decay=cosim_decay, use_sgdm=use_sgdm,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or const != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero const")
        super(Adabeta, self).__init__(params, defaults)
        self.resetOfflineStats()
        self.cosim_list = []

    def __setstate__(self, state):
        super(Adabeta, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def getOfflineStats(self):
        return self.offline_grad

    def resetOfflineStats(self):
        self.offline_grad = {'yes': 0, 'no': 0}

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.cosim = 0.0
        self.cosim_grad = 0.0
        self.momentum = np.nan

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            const = group['const']
            cosim_decay = group['cosim_decay']
            nesterov = group['nesterov']
            use_sgdm = group['use_sgdm']

            # grad_norm = 0.0
            params = []
            moms = []
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'momentum_buffer' in param_state:
                    buf = param_state['momentum_buffer']
                # grad_norm += torch.sqrt(torch.sum(torch.square(d_p)))
                    params += [p.grad.data.view(-1)]
                    moms += [buf.view(-1)]
            if len(moms)>0:
                self.cosim = torch.nn.CosineSimilarity(dim=0)(torch.cat(params), torch.cat(moms))

            try:
              self.cosim_grad = torch.nn.CosineSimilarity(dim=0)(torch.cat(params), torch.cat(self.params))
            except:
              self.cosim_grad = 0
            self.params = params
            self.momentum = moms

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                self.old_grad = d_p
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    param_state['cosim_run_avg'] = 0.0
                    param_state['step'] = 0
                else:
                    buf = param_state['momentum_buffer']
                    if not use_sgdm:
                        cosim_avg = param_state['cosim_run_avg']
                        if cosim_decay != 0.0:
                            cosim_avg = cosim_avg * cosim_decay + self.cosim * (1-cosim_decay)
                        else:
                            cosim_avg = self.cosim
                        dampening = const + (cosim_avg + 1) / 2
                        param_state['cosim_run_avg'] = cosim_avg
                    else:
                        dampening = 1.0
                    buf.mul_(momentum).add_(d_p, alpha=dampening)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf
                param_state['step'] += 1
                p.data.add_(d_p, alpha=-group['lr'])
                self.update_mean = d_p

        return loss



