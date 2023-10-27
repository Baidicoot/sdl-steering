import copy

import optree
import torch
import torch.nn.functional as F
from torch.func import stack_module_state, functional_call
import torchopt

class Ensemble:
    def __init__(
        self,
        models,
        optimizer_func,
        optimizer_kwargs,
        model_hyperparams,
        device=None,
        no_stacking=False,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_models = len(models)
        self.params, self.buffers = stack_module_state(models)

        self.sig = copy.deepcopy(models[0]).to("meta")
        self.no_stacking = no_stacking

        self.optimizer_func = optimizer_func
        self.optimizer_kwargs = optimizer_kwargs

        self.optimizer = optimizer_func(**optimizer_kwargs)
        self.optim_states = torch.vmap(self.optimizer.init)(self.params)

        self.model_hyperparams = model_hyperparams

        self.init_functions()

    def init_functions(self):
        def call_single_model(params, buffers, batch):
            outputs = functional_call(self.sig, (params, buffers), batch)
            return outputs[0], outputs

        def calc_grads(params, buffers, batch):
            return torch.func.grad(call_single_model, has_aux=True)(params, buffers, batch)

        self.calc_grads = torch.vmap(calc_grads)
        self.update = torch.vmap(self.optimizer.update)

    def unstack(self, device=None):
        for i in range(self.n_models):
            model = copy.deepcopy(self.sig).to(device)
            state_dict = {}
            state_dict.extend({k: p[i] for k, p in self.params.items()})
            state_dict.extend({k: p[i] for k, p in self.buffers.items()})
            model.load_state_dict(state_dict)
            yield model

    def to_device(self, device):
        self.device = device

        for t in self.params.values():
            t.to(device)
        
        for t in self.buffers.values():
            t.to(device)
        
        for t in self.optim_states.values():
            t.to(device)

    def step_batch(self, minibatches, expand_dims=True):
        with torch.no_grad():
            if expand_dims:
                minibatches = minibatches.expand(self.n_models, *minibatches.shape)

            grads, outputs = self.calc_grads(self.params, self.buffers, minibatches)

            updates, new_optim_states = self.update(grads, self.optim_states)

            self.optim_states = new_optim_states

            torchopt.apply_updates(self.params, updates)

            return outputs