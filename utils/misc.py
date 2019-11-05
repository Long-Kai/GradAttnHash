import torch
import os


class GPU(object):
    def __init__(self, device="cuda:0"):
        self.device = torch.device(device)

    def autocuda(self, var):
        # use_cuda = True
        # device = torch.device("cuda:0" if use_cuda else "cpu")
        if torch.cuda.is_available():
            return var.to(self.device)
        else:
            return var


def convert_to_onehot(labels, num_class):
    """labels is a one-dimensional Int Tensor"""

    y_onehot = torch.LongTensor(len(labels), num_class)
    y_onehot.zero_()
    y_onehot.scatter_(1, torch.unsqueeze(labels.cpu(), 1), 1)
    return y_onehot


def save_model(model, save_model_to):
    """models is a dict {name:model_obj}"""
    torch.save(model, os.path.join(save_model_to, "{}.model".format("model")))


def load_models(path, test_mode=True):
    """load .model files, return a dict {name:model_obj}"""
    return torch.load(os.path.join(path, "{}.model".format("model")))


def load_dict(model, dict):
    model_dict = model.state_dict()
    dict = {k: v for k, v in dict.items() if k in model_dict}
    model_dict.update(dict)
    model.load_state_dict(model_dict)
    return model

