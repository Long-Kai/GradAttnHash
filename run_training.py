import os
from utils.misc import GPU
from utils.train_utils import get_common_config
from train_grad_attn import train

# config

gpu_id = 0
if gpu_id is None:
    device = "cpu"
elif gpu_id == 0:
    device = "cuda:0"
else:
    device = "cuda:1"

gpu = GPU(device)

ds = "imagenet"  # dataset

tr_config = get_common_config(ds)

tr_config["num_bits"] = 64

tr_config["autocuda"] = gpu.autocuda

# default param: sigmoid_param, init_lr, attn_lr
# cifar 16/32: 0.2, 0.001, 0.08
# cifar 48/64: 0.2, 0.0005, 0.05
# nuswide: 0.3, 0.001, 0.1
# imagenet: 0.15, 0.001, 0.08

tr_config["loss"]["sigmoid_param"] = 0.15   # \gamma in eq (8)
tr_config["optimizer"]["lr_param"]["init_lr"] = 0.001   # learning rate of original hash network
tr_config["attn_lr"] = 0.08    # learning rate of grad attn net

tr_config["save_model"] = False
if tr_config["save_model"]:
    tr_config["output_path"] = "./saved_model/" + tr_config["dataset"] + "_" + str(
        tr_config["num_bits"]) + "bit_GradAttnHash"
    if not os.path.exists(tr_config["output_path"]):
        os.mkdir(tr_config["output_path"])

# print(tr_config)

print("Training starts")
train(tr_config)
