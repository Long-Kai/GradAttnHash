import torch
from torch.autograd import Variable
import math


def preprocess_gradients(x):
    """to reduce the range of gradient input"""
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)


def reset_model(model):

    for name, param in model.named_parameters():
        if len(name.split('.')) == 3:
            model.__getattr__(name.split('.')[0])[int(name.split('.')[1])]._parameters[
                name.split('.')[2]] = Variable(param.data)
        elif len(name.split('.')) == 2:
            model.__getattr__(name.split('.')[0])._parameters[name.split('.')[1]] = Variable(param.data)


def copy_params(model_from, model_to):
    # for modelA, modelB in zip(model_from.parameters(), model_to.parameters()):
    #     modelB.data.copy_(modelA.data)
    for (Aname, modelA), (Bname, modelB) in zip(model_from.named_parameters(), model_to.named_parameters()):
        # print(Aname, Bname)
        modelB.data.copy_(modelA.data)


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)

    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1

    return optimizer


def step_lr_scheduler(param_lr, optimizer, iter_num, gamma, step, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (gamma ** (iter_num // step))

    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1

    return optimizer


def multistep_lr_scheduler(param_lr, optimizer, init_lr, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = init_lr
    elif epoch < 120:
        lr = init_lr * 0.1
    else:
        lr = init_lr * 0.01

    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer


schedule_dict = {"inv": inv_lr_scheduler, "step": step_lr_scheduler, "multistep": multistep_lr_scheduler}


def get_common_config(ds):
    config = {"dataset": ds}
    batch_size = 128

    if config["dataset"] == "nuswide":
        positive_weight = 5
        top_return = 5000
        config["data"] = {"train_set": {"list_path": "./data/nuswide_81/train.txt",
                                        "batch_size": batch_size, "dataset": config["dataset"]},
                          "db_set": {"list_path": "./data/nuswide_81/database.txt",
                                     "batch_size": batch_size, "dataset": config["dataset"]},
                          "query_set": {"list_path": "./data/nuswide_81/test.txt",
                                        "batch_size": batch_size, "dataset": config["dataset"]}}
        config["test_10crop"] = True
        config["optimizer"] = {"type": "SGD", "optim_params": {"lr": 1.0, "momentum": 0.9,
                                                               "weight_decay": 0.0005, "nesterov": True},
                               "lr_type": "step",
                               "lr_param": {"gamma": 0.5, "step": 79 * 30}}
        config["test_period"] = 79 * 5

    elif config["dataset"] == "imagenet":
        positive_weight = 100
        top_return = 1000
        config["data"] = {"train_set": {"list_path": "./data/imagenet/train.txt",
                                        "batch_size": batch_size, "dataset": config["dataset"]},
                          "db_set": {"list_path": "./data/imagenet/database.txt",
                                     "batch_size": batch_size, "dataset": config["dataset"]},
                          "query_set": {"list_path": "./data/imagenet/test.txt",
                                        "batch_size": batch_size, "dataset": config["dataset"]}}
        config["test_10crop"] = True
        config["optimizer"] = {"type": "SGD", "optim_params": {"lr": 1.0, "momentum": 0.9,
                                                               "weight_decay": 0.0005, "nesterov": True},
                               "lr_type": "step",
                               "lr_param": {"gamma": 0.5, "step": 102 * 80}}
        config["test_period"] = 102 * 5

    elif config["dataset"] == "cifar":
        positive_weight = 10
        top_return = 5000
        config["data"] = {"train_set": {"list_path": "./data/cifar/train.txt",
                                        "batch_size": batch_size, "dataset": config["dataset"]},
                          "db_set": {"list_path": "./data/cifar/database.txt",
                                     "batch_size": batch_size, "dataset": config["dataset"]},
                          "query_set": {"list_path": "./data/cifar/test.txt",
                                        "batch_size": batch_size, "dataset": config["dataset"]}}
        config["test_10crop"] = False
        config["optimizer"] = {"type": "SGD", "optim_params": {"lr": 1.0, "momentum": 0.9,
                                                               "weight_decay": 0.0005, "nesterov": True},
                               "lr_type": "step",
                               "lr_param": {"gamma": 0.5, "step": 40 * 30}}  # 40 iter = 1 epoch
        config["test_period"] = 40 * 5

    config["loss"] = {"positive_weight": positive_weight}

    config["test"] = {"top_return": top_return}

    config["epoch"] = 301

    return config

