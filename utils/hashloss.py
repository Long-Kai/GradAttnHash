"""
Utilities of loss for learning to hash
"""

import os
import sys
import torch
from torch.autograd import Variable


sys.path.append(os.path.dirname(__file__))


def sim_log_loss_double(feats1, labels1, feats2, labels2, positive_weight=1, normalize=True, sigmoid_param=0.5, hook=None):

    # do inner product of features
    predicted_sim = torch.mm(feats1, feats2.t())
    if hook is not None:
        predicted_sim.register_hook(hook)
    predicted_sim = predicted_sim * sigmoid_param
    true_sim = Variable(torch.mm(labels1.data.float(), labels2.data.float().t()) > 0).float()
    weighted_matrix = (positive_weight - 1) * true_sim + 1

    log_prob = torch.log(1 + torch.exp(predicted_sim)) - torch.mul(true_sim, predicted_sim)
    log_prob = torch.mul(log_prob, weighted_matrix)
    sum_log_prob = log_prob.sum()

    # divide the sum to get average loss
    return sum_log_prob / torch.sum(weighted_matrix) if normalize else sum_log_prob




