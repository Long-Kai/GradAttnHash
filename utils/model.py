import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.optim import Optimizer
from utils.train_utils import preprocess_gradients


class MyDropout(nn.Module):

    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training or not self.p:
            return x

        if self.mask is None:
            self.mask = x.new_empty(x.size(), requires_grad=False).bernoulli_(1 - self.p) / (1 - self.p)
        return x * self.mask

    def clear_mask(self):
        self.mask = None


class MyAlexNet(nn.Module):

    def __init__(self, num_bits=32):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            MyDropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            MyDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.feature_layers = nn.Sequential(self.features, self.classifier)
        self.hash_layer = nn.Linear(4096, num_bits)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.hash_layer(x)  # tanh is removed
        return x

    def clear_dropout_mask(self):
        self.classifier[0].clear_mask()
        self.classifier[3].clear_mask()

    def print_dropout_mask(self):
        print(self.classifier[0].mask)
        print(self.classifier[3].mask)


class AlexNet(nn.Module):

    def __init__(self, num_bits=32):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.feature_layers = nn.Sequential(self.features, self.classifier)
        self.hash_layer = nn.Linear(4096, num_bits)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        # x.register_hook(lambda grad: grad.detach())
        x = self.hash_layer(x)  # tanh is removed
        return x


def copy_mask_MyAlexNet(model_from, model_to):
    model_to.classifier[0].mask = model_from.classifier[0].mask.clone()
    model_to.classifier[3].mask = model_from.classifier[3].mask.clone()


def pretrain_alexnet(model):
    model_dict = model.state_dict()
    model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }
    pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


class AttnNet(nn.Module):
    def __init__(self, hidden_size=100):
        super(AttnNet, self).__init__()
        self.fc_in = nn.Linear(5, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        x = F.relu(self.fc_in(inputs))
        x = self.fc_out(x)
        return x

    def get_sim_grad(self, grads):
        """
                use as hook to change the grad of output of network;
                org_grads: use to get original grad, no use in the func
                sim_grads: grad of the predicted sim mat/ codes inner product mat; obtained by hook; (n x n)
                cont_codes: continuous code (n x k)
                code_grads: grad of codes of each instance (n x k)
        """

        sim_grads = grads.data.clone()

        # hook function, the inputs are from the obj member;
        # must run set_cont_codes and set_sim_grads (as hook) before calling this function
        cont_codes_1 = self.cont_codes_1
        cont_codes_2 = self.cont_codes_2

        code_len = cont_codes_1.size(1)
        batch_size = cont_codes_1.size(0)
        code_grads_1 = torch.matmul(sim_grads, cont_codes_2)
        code_grads_2 = torch.matmul(sim_grads.t(), cont_codes_1)
        # print("code_grad_1", code_grads_1.data)
        # print("code_grad_2", code_grads_2.data)

        # n*n x k; b1,b1,...,b1|b2,b2..
        cont_codes_input_one_by_one = cont_codes_1.repeat(1, batch_size).view(-1, code_len)

        # n*n x k; a1,a2,..,an|a1,a2,..,an
        cont_codes_input_batch_by_batch = cont_codes_2.repeat(batch_size, 1).view(-1, code_len)

        # repeated grad of sim mat to n*n x k
        # s11, s12, ... s1n| s21, s22, ....
        sim_grads_one_by_one = sim_grads.view(-1, 1).repeat(1, code_len).view(-1, code_len)

        # compute code grad of each pairwise loss
        pairwise_grads_1 = torch.mul(cont_codes_input_batch_by_batch, sim_grads_one_by_one).view(-1, 1)
        pairwise_grads_2 = torch.mul(cont_codes_input_one_by_one, sim_grads_one_by_one).view(-1, 1)

        # print('test_pair_grad_1', pairwise_grads_1.view(batch_size, batch_size, code_len).sum(1))
        # print('test_pair_grad_2', pairwise_grads_2.view(batch_size, batch_size, code_len).sum(0))

        # compute code grad against all other codes
        batchwise_grads_1 = code_grads_1.repeat(1, batch_size).view(-1, code_len).view(-1, 1)
        batchwise_grads_2 = code_grads_2.repeat(batch_size, 1).view(-1, code_len).view(-1, 1)

        inputs = torch.cat((cont_codes_input_one_by_one.view(-1, 1), preprocess_gradients(pairwise_grads_1),
                            preprocess_gradients(batchwise_grads_1)), 1)
        weights_1 = self(inputs)

        inputs = torch.cat((cont_codes_input_batch_by_batch.view(-1, 1), preprocess_gradients(pairwise_grads_2),
                            preprocess_gradients(batchwise_grads_2)), 1)
        weights_2 = self(inputs)

        attn = F.softmax(torch.cat((weights_1, weights_2), 1), dim=1)

        attn_1 = attn[:, 0]
        mask = attn_1.view(batch_size * batch_size, code_len)
        weighted_code_grad_1 = torch.mul(cont_codes_input_batch_by_batch, sim_grads_one_by_one) * 2
        weighted_code_grad_1 = torch.mul(weighted_code_grad_1, mask)
        weighted_code_grad_1 = weighted_code_grad_1.view(batch_size, batch_size, code_len).sum(1)

        attn_2 = attn[:, 1]
        mask = attn_2.view(batch_size * batch_size, code_len)
        weighted_code_grad_2 = torch.mul(cont_codes_input_one_by_one, sim_grads_one_by_one) * 2
        weighted_code_grad_2 = torch.mul(weighted_code_grad_2, mask)
        weighted_code_grad_2 = weighted_code_grad_2.view(batch_size, batch_size, code_len).sum(0)

        self.weighted_code_grad_1 = weighted_code_grad_1
        self.weighted_code_grad_2 = weighted_code_grad_2
        self.weight_loss = torch.mean(torch.pow(attn - 0.5, 2))



    def get_cont_code_1(self, cont_codes):
        self.cont_codes_1 = cont_codes.detach().clone()

    def get_cont_code_2(self, cont_codes):
        self.cont_codes_2 = cont_codes.detach().clone()

    def set_code_grad_1(self, org_grads):
        # print("org_grad_1", org_grads)
        return self.weighted_code_grad_1

    def set_code_grad_2(self, org_grads):
        # print("org_grad_2", org_grads)
        return self.weighted_code_grad_2



class SGDWoReplace(Optimizer):

    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDWoReplace, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDWoReplace, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, model_with_grads, model_to_update):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        model_with_grads_iter = model_with_grads.named_parameters()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p, (qname, q) in zip(group['params'], model_with_grads_iter):
                if len(qname.split('.')) == 3:
                    var = model_to_update.__getattr__(qname.split('.')[0])[int(qname.split('.')[1])]._parameters[
                        qname.split('.')[2]]
                elif len(qname.split('.')) == 2:
                    var = model_to_update.__getattr__(qname.split('.')[0])._parameters[qname.split('.')[1]]


                if q.grad is None:
                    continue
                # d_p = p.grad.data
                d_p = q.grad.clone()
                if weight_decay != 0:
                    # d_p.add_(weight_decay, p.data)
                    d_p = d_p.add(weight_decay, var)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        # buf.mul_(momentum).add_(d_p)
                        buf = buf.mul(momentum).add(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_(1 - dampening, d_p)
                        buf = buf.mul(momentum).add(1 - dampening, d_p)
                        param_state['momentum_buffer'].data.copy_(buf.data)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                if len(qname.split('.')) == 3:
                    model_to_update.__getattr__(qname.split('.')[0])[int(qname.split('.')[1])]._parameters[
                        qname.split('.')[2]] = torch.add(var, -group['lr'], d_p)
                elif len(qname.split('.')) == 2:
                    model_to_update.__getattr__(qname.split('.')[0])._parameters[qname.split('.')[1]] = \
                        torch.add(var, -group['lr'], d_p)
                # p.data.add_(-group['lr'], d_p)






