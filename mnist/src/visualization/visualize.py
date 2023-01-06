import torch

model = torch.load(model_checkpoint)

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
