
# Requires model to be an instance of nn.Module
def num_of_params(model):
    return sum(p.numel() for p in model.parameters())