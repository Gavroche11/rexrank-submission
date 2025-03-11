from rexrank.classifier.eva_x import build_eva_x

import torch.nn as nn

def load_model(model_name: str,
               **kwargs) -> nn.Module:
    if model_name == "eva-x":
        model = build_eva_x(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model