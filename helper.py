import torch

def learnable_params(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    print("Model’s state dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("Optimizer’s state dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])