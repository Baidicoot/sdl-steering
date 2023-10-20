import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import training.dictionary as sae
import torch

def compare_models_ensemble_output(models, train_step, batch):
    _, _, outputs = train_step(batch, opt_step=False, return_output=True)
    outputs = outputs["output"]

    model_outputs = []

    for model in models:
        x_hat, _, _, _ = model(batch)

        model_outputs.append(x_hat.detach())

    diff = torch.empty(len(models)).to("cpu")
    param_diff = {}

    for i in range(len(models)):
        diff[i] = torch.mean((outputs[i] - model_outputs[i])**2).to("cpu")

        for name, param in models[i].named_parameters():
            if name not in param_diff:
                param_diff[name] = []
            
            param_diff[name].append(torch.mean((params[name][i].detach() - param.data.detach())**2).to("cpu"))

    return diff, param_diff

if __name__ == "__main__":
    train_step, unstack_models, params, buffers = sae.make_ensemble(10, 10, [0.1, 0.2, 0.3], {"lr": 3e-4})

    batch = torch.randn(100, 10).to("cpu")

    models = unstack_models(params, buffers, "cpu")
    diff, param_diff = compare_models_ensemble_output(models, train_step, batch)

    print(diff, param_diff)

    batch = torch.randn(100, 10).to("cpu")

    for i in range(10000):
        train_step(batch)

    batch = torch.randn(100, 10).to("cpu")

    diff, param_diff = compare_models_ensemble_output(models, train_step, batch)

    print(diff, param_diff)

    batch = torch.randn(100, 10).to("cpu")

    models = unstack_models(params, buffers, "cpu")
    diff, param_diff = compare_models_ensemble_output(models, train_step, batch)

    print(diff, param_diff)