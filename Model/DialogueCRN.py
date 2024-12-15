import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from vision_lstm import ViLBlock

def pad(tensor, length, cuda_flag):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if cuda_flag:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if cuda_flag:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def feature_transfer(bank_s_, bank_p_, seq_lengths, cuda_flag=False):
    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    if cuda_flag:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)
    # (l,b,h)
    bank_s = torch.stack(
        [pad(bank_s_.narrow(0, s, l), max_len, cuda_flag) for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1)
    bank_p = torch.stack(
        [pad(bank_p_.narrow(0, s, l), max_len, cuda_flag) for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1)

    return bank_s, bank_p

class simam_module(torch.nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, l = x.size()  # 适应三维输入

        n = l - 1

        # 计算 x 减去均值的平方
        x_minus_mu_square = (x - x.mean(dim=2, keepdim=True)).pow(2)

        # 计算 y
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=2, keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)

class ReasonModule(nn.Module):
    def __init__(self, in_channels=200, processing_steps=0, num_layers=1):
        """
        Reasoning Module
        """
        super(ReasonModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        if processing_steps > 0:
            self.xlstm = ViLBlock(self.out_channels, self.in_channels, num_layers)  # 400,200,1
            self.xlstm.reset_parameters()
        self.simam = simam_module(in_channels)

    def forward(self, x, batch, q_star):
        if self.processing_steps <= 0: return q_star

        batch_size = batch.max().item() + 1
        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        for i in range(self.processing_steps):
            q, h = self.xlstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            q = self.simam(q)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)
        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class DialogueCRN(nn.Module):
    def __init__(self, n_features=200, n_classes=7, dropout=0.2, cuda_flag=False, reason_steps=None):
        """
        Multi-turn Reasoning Modules
        """
        super(DialogueCRN, self).__init__()
        self.cuda_flag = cuda_flag
        self.fc = nn.Linear(n_features, n_features * 2)
        self.steps = reason_steps if reason_steps is not None else [0, 0]
        self.reason_modules = nn.ModuleList([
            ReasonModule(in_channels=n_features, processing_steps=self.steps[0], num_layers=1),
            ReasonModule(in_channels=n_features, processing_steps=self.steps[1], num_layers=1)
        ])
        self.dropout = nn.Dropout(dropout)
        self.smax_fc = nn.Linear(n_features * 4, n_classes)

    def forward(self, U_s, U_p, seq_lengths):
        # (b) <== (l,b,h)
        batch_size = U_s.size(1)
        batch_index, context_s_, context_p_ = [], [], []
        for j in range(batch_size):
            batch_index.extend([j] * seq_lengths[j])
            context_s_.append(U_s[:seq_lengths[j], j, :])
            context_p_.append(U_p[:seq_lengths[j], j, :])

        batch_index = torch.tensor(batch_index)
        bank_s_ = torch.cat(context_s_, dim=0)
        bank_p_ = torch.cat(context_p_, dim=0)
        if self.cuda_flag:
            batch_index = batch_index.cuda()
            bank_s_ = bank_s_.cuda()
            bank_p_ = bank_p_.cuda()

        # (l,b,h) << (l*b,h)
        bank_s, bank_p = feature_transfer(bank_s_, bank_p_, seq_lengths, self.cuda_flag)

        feature_ = []
        for t in range(bank_s.size(0)):
            # (2*h) <== (h)
            q_star = self.fc(bank_s[t])
            q_situ = self.reason_modules[0](bank_s_, batch_index, q_star)
            feature_.append(q_situ.unsqueeze(0))
        feature_s = torch.cat(feature_, dim=0)

        feature_ = []
        for t in range(bank_p.size(0)):
            q_star = self.fc(bank_p[t])
            q_party = self.reason_modules[1](bank_p_, batch_index, q_star)
            feature_.append(q_party.unsqueeze(0))
        feature_v = torch.cat(feature_, dim=0)

        # (l,b,2*2*h)
        hidden = torch.cat([feature_v, feature_s], dim=-1)
        hidden = self.dropout(F.relu(hidden))
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob