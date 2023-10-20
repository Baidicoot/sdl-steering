import torch
import torch.nn as nn

import copy

from torch.func import stack_module_state, vmap, functional_call

class SparseLinearAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, l1_penalty):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)

        # initialize biases
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        
        # have l1 penalty as a buffer
        self.register_buffer('l1_penalty', torch.tensor(l1_penalty))
    
    def forward(self, x):
        c = self.encoder(x)
        c = torch.relu(c)

        # normalize decoder weights
        w = self.decoder.weight
        w = w / torch.norm(w, dim=1, keepdim=True)
        x_hat = torch.matmul(c, w.t()) + self.decoder.bias

        reconstruction_loss = torch.mean((x - x_hat)**2)
        l1_loss = torch.sum(c)

        loss = reconstruction_loss + self.l1_penalty * l1_loss

        return x_hat, loss, reconstruction_loss, c

def make_ensemble(input_dim, hidden_dim, l1_range, adam_settings):
    # create a list of models
    models = []
    for l1_penalty in l1_range:
        models.append(SparseLinearAutoencoder(input_dim, hidden_dim, l1_penalty))

    params, buffers = stack_module_state(models)

    model_meta = copy.deepcopy(models[0])
    model_meta.to("meta")

    n_models = len(models)

    def call_single_model(params, buffers, batch):
        return functional_call(model_meta, (params, buffers), batch)

    optimizer = torch.optim.Adam(params.values(), **adam_settings)

    def train_step(batch, expand_batch=True, opt_step=True, return_output=False, return_hidden=False):
        if expand_batch:
            batch = batch.expand(n_models, -1, -1)

        if opt_step:
            optimizer.zero_grad()

        x_hat, loss, reconstr_loss, c = vmap(call_single_model)(params, buffers, batch)
        
        if opt_step:
            loss.sum().backward()
            #print(loss.sum().item())
            optimizer.step()

        extra_output = {}
        if return_output:
            extra_output["output"] = x_hat.detach()
        
        if return_hidden:
            extra_output["hidden"] = c.detach()

        return loss.detach(), reconstr_loss.detach(), extra_output
    
    def unstack_models(params, buffers, device):
        models = []
        
        for i in range(n_models):
            model = SparseLinearAutoencoder(input_dim, hidden_dim, l1_range[i]).to(device)

            for name, param in model.named_parameters():
                param.data = params[name][i].clone().detach().to(device)
            
            models.append(model)
        
        return models
    
    return train_step, unstack_models, params, buffers