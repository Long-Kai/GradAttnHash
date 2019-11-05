import torch.optim as optim
import torch
import math
import os
import sys
from utils.datasets import get_train_data, get_test_query, get_test_db
from utils.model import MyAlexNet, pretrain_alexnet, SGDWoReplace, AttnNet, copy_mask_MyAlexNet
from utils.misc import load_dict
from utils.train_utils import copy_params, reset_model, schedule_dict
from utils.test_utils import run_test
from utils.hashloss import sim_log_loss_double

sys.path.append(os.path.dirname(__file__))


def train(config):
    autocuda = config["autocuda"]

    test_period = config["test_period"]

    init_scale = 1.0
    scale = init_scale
    scale_step_size = 200
    scale_iter_num = 0
    scale_gamma = 0.005
    scale_power = 0.5

    train_loader_1 = get_train_data(config["data"]["train_set"])
    train_loader_2 = get_train_data(config["data"]["train_set"], seed=3571)

    query_loader = get_test_query(config["data"]["query_set"], config["test_10crop"])
    db_loader = get_test_db(config["data"]["db_set"], config["test_10crop"])

    net = pretrain_alexnet(autocuda(MyAlexNet(num_bits=config["num_bits"])))
    net.train(True)
    net_shadow = autocuda(MyAlexNet(num_bits=config["num_bits"]))
    copy_params(net, net_shadow)
    attn_net = autocuda(AttnNet(hidden_size=100))
    attnnet_optimizer = optim.Adam(attn_net.parameters(), lr=config["attn_lr"])

    # torch.save(net.hash_layer.state_dict(), os.path.join("./saved_model/init/", "hash_layer64".format("model")))
    # torch.save(net.hash_layer.state_dict(), os.path.join("./saved_model/init/", "hash_layer64".format("model")))
    # torch.save(attn_net.state_dict(), os.path.join("./saved_model/init/", "attn".format("model")))

    hash_layer_dict = torch.load(os.path.join("./saved_model/init/", ("hash_layer"
                                                                       + str(config["num_bits"])).format("model"))
                                 , map_location=lambda storage, loc: storage)
    attn_net_dict = torch.load(os.path.join("./saved_model/init/", "attn".format("model")),
                               map_location=lambda storage, loc: storage)

    load_dict(net.hash_layer, hash_layer_dict)
    load_dict(attn_net, attn_net_dict)

    optimizer_config = config["optimizer"]
    parameter_list = [{"params": net_shadow.feature_layers.parameters(), "lr": 1},
                      {"params": net_shadow.hash_layer.parameters(), "lr": 10}]
    optimizer = SGDWoReplace(parameter_list, **(optimizer_config["optim_params"]))

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = schedule_dict[optimizer_config["lr_type"]]
    j = 0
    running_loss = 0.0
    running_attn_loss = 0.0

    for epoch in range(config["epoch"]):  # loop over the dataset multiple times

        for i, ((data1, labels1), (data2, labels2)) in enumerate(zip(train_loader_1, train_loader_2), 0):

            net.train()
            net_shadow.train()

            data1, labels1 = autocuda(data1), autocuda(labels1)
            data2, labels2 = autocuda(data2), autocuda(labels2)

            # zero the parameter gradients
            optimizer = lr_scheduler(param_lr, optimizer, j + i, **schedule_param)
            net.zero_grad()
            net.clear_dropout_mask()
            reset_model(net_shadow)
            attnnet_optimizer.zero_grad()

            scale_iter_num += 1
            if scale_iter_num % scale_step_size == 0:
                scale = init_scale * math.pow((1. + scale_gamma * scale_iter_num), scale_power)

            # forward + backward + optimize
            batch_size = data1.size(0)
            data = torch.torch.cat((data1, data2), dim=0)
            feat = net(data)
            feats1, feats2 = torch.split(feat, batch_size, dim=0)
            outputs1 = torch.tanh(scale * feats1)
            outputs2 = torch.tanh(scale * feats2)

            attn_net.get_cont_code_1(outputs1)
            attn_net.get_cont_code_2(outputs2)
            h1 = outputs1.register_hook(attn_net.set_code_grad_1)
            h2 = outputs2.register_hook(attn_net.set_code_grad_2)
            loss = sim_log_loss_double(outputs1, labels1, outputs2, labels2,
                                       positive_weight=config["loss"]["positive_weight"],
                                       sigmoid_param=config["loss"]["sigmoid_param"],
                                       hook=attn_net.get_sim_grad)

            loss.backward(create_graph=True)
            optimizer.step(net, net_shadow)

            copy_params(net_shadow, net)

            h1.remove()
            h2.remove()

            if epoch <= 1000:
                copy_mask_MyAlexNet(net, net_shadow)
                feat = net_shadow(data)
                feats1, feats2 = torch.split(feat, batch_size, dim=0)

                outputs3 = torch.tanh(scale * feats1)
                outputs4 = torch.tanh(scale * feats2)

                attn_loss = sim_log_loss_double(outputs3, labels1, outputs4, labels2,
                                                sigmoid_param=config["loss"]["sigmoid_param"],
                                                positive_weight=config["loss"][
                                                    "positive_weight"])

                attn_loss.backward()
                attnnet_optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_attn_loss += attn_loss.item()
            if (i + j) % test_period == (test_period - 1):

                m_a_p= run_test(query_loader, db_loader, net.eval(),autocuda, test_10crop=config["test_10crop"],
                                                                          num_query=1000,
                                                                          num_top_return=config["test"]["top_return"])
                print('[%d, %5d] loss: %.3f; loss: %.3f; MAP: %.4f' %
                      (epoch + 1, i + 1, running_loss / test_period, running_attn_loss / test_period, m_a_p))

                if config["save_model"]:
                    torch.save(net.state_dict(), os.path.join(config["output_path"], "alexnet".format("model")))

                running_loss = 0.0
                running_attn_loss = 0.0
                net.train(True)
        j = j + i + 1

    print('Training is complete')



