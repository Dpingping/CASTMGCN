import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import math
import numpy as np


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        if tuple(A.shape) == (1, 207, 207):
            A = A.squeeze()
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=4, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        # c_in = (order * support_len + 1) * c_in
        # 32 * (4 - 1),32 * (4+1)
        self.mlp = linear(c_in * (support_len - 1), c_out)

        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for i, a in enumerate(support):
            a = a.squeeze()
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class AttenSum(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttenSum, self).__init__()
        self.Q = nn.Linear(input_size, output_size, bias=False)  # Define parameter Q
        self.R = nn.Linear(output_size, output_size, bias=False)  # Define parameter R
        self.b = nn.Parameter(torch.randn(output_size))  # Define parameter b

    def forward(self, *Z_list):
        betas = []

        for Z in Z_list:
            batch_size, T, nodes, features = Z.shape
            Z = Z.reshape(batch_size * T, nodes, features)  # Reshape to [64*12, 617, 32]

            QZ_plus_b = self.Q(Z) + self.b.unsqueeze(0).unsqueeze(0)  # Calculate Q·Z+b, add batch dimension
            q_i = torch.matmul(QZ_plus_b, self.R.weight.t())  # Calculate Q·Z+b·R^T
            beta = torch.softmax(q_i, dim=1)  # Calculate beta
            beta = beta.reshape(batch_size, T, nodes)  # Reshape to [64, 12, 617]
            betas.append(beta)
        res = []
        for i, beta in enumerate(betas):
            beta_expanded = beta.unsqueeze(-1)  # Expand the last dimension of w to match the last dimension of x
            weighted_x = Z_list[i] * beta_expanded  # Weight x
            res.append(weighted_x)
        accumulated_result = torch.sum(torch.stack(res, dim=0), dim=0).permute(0, 3, 2, 1)
        return accumulated_result


def InterviewAttention(V, H):
    # V[64, 2468, 12] H [64, 128, 617, 12]
    V = V.reshape(64, 4, 207, 12)
    V_attn = nn.Linear(V.shape[1], V.shape[1])(V)

    V_attn = nn.ReLU()(V_attn)
    V_attn = nn.Linear(V_attn.shape[1], V_attn.shape[1])(V_attn)
    V_attn = nn.Sigmoid()(V_attn)
    dot_product = torch.einsum('bij,bj->bi', H, V_attn)
    return dot_product


def AttenPoolSum(H_1, H_2, H_3, H_4):
    H = []
    # print('torch.mean(H_1, dim=1)', torch.mean(H_1, dim=1).shape)
    z = torch.cat((torch.mean(H_1, dim=1), torch.mean(H_2, dim=1), torch.mean(H_3, dim=1), torch.mean(H_4, dim=1)),
                  dim=1)
    # Concatenate each view to Ht
    Ht = torch.cat((H_1, H_2, H_3, H_4), dim=1)
    print('Ht', Ht.shape)  # [64, 128, 617, 12] ------> [64, 32，4, 617, 12]
    print('z', z.shape)  # [64, 2468, 12] z [64,4,617, 12]
    H.append(InterviewAttention(z, Ht))  # get scaled Ht


class MultiGCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_list=[6, 6, 6, 4], order=2, fusion='mean'):
        super(MultiGCN, self).__init__()
        self.mgcn = nn.ModuleList()
        self.weightedsum = WeightedSumModel()
        self.AttenSum = AttenSum(c_in, 1)
        self.fusion = fusion
        self.max_selections = [0] * len(support_list)  # Used for max selection statistics
        for i in range(len(support_list)):
            self.mgcn.append(gcn(c_in, c_out, dropout, support_len=support_list[i], order=order))

    def forward(self, x, supports):
        result = []
        for i, adj in enumerate(supports):
            result.append(self.mgcn[i](x, adj))  # [64, 32, 617, 12,4]
        if self.fusion == 'sum':
            feat_fusion = torch.sum(torch.stack(result, dim=-1), dim=-1)
        elif self.fusion == 'weightsum':
            feat_fusion = self.weightedsum(result)
        elif self.fusion == 'mean':
            feat_fusion = torch.mean(torch.stack(result, dim=-1), dim=-1)
        # Absolute value
        elif self.fusion == 'max':
            stacked = torch.stack(result, dim=-1)
            # print('stacked', stacked.shape)
            # max_values, max_indices = torch.max(stacked, dim=-1)
            # Take the maximum absolute value
            abs_stacked = torch.abs(stacked)
            max_values, max_indices = torch.max(abs_stacked, dim=-1)

            # print('max_values', max_values.shape)
            # Count how many times each matrix is selected
            for i in range(len(supports)):
                self.max_selections[i] += (max_indices == i).sum().item()
            feat_fusion = max_values
        elif self.fusion == 'atten':
            tensor1, tensor2, tensor3, tensor4 = result[0].permute(0, 3, 2, 1), result[1].permute(0, 3, 2, 1), \
                result[2].permute(0, 3, 2, 1), result[3].permute(0, 3, 2, 1)
            feat_fusion = self.AttenSum(tensor1, tensor2, tensor3, tensor4)

        return feat_fusion

    def get_max_selection_stats(self):
        total = sum(self.max_selections)
        if total == 0:
            return "Max fusion method was not used"
        percentages = [count / total * 100 for count in self.max_selections]
        return {
            "total_matrices": len(self.max_selections),
            "selections": self.max_selections,
            "percentages": percentages
        }


class WeightedSumModel(nn.Module):
    def __init__(self):
        super(WeightedSumModel, self).__init__()
        self.weights = nn.Parameter(torch.ones(4, requires_grad=True))

    def forward(self, tensor_list):
        weighted_tensors = [tensor * weight for tensor, weight in zip(tensor_list, self.weights)]
        weighted_sum = torch.stack(weighted_tensors).sum(dim=0)
        return weighted_sum


# temporal attention

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size, device):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)
        self.device = device

    def forward(self, input):
        # input is dim (N, in_channels, T) where N is the batch_size, and T is the sequence length
        mask = np.triu(np.ones((input.size(2), input.size(2))), k=1).astype(bool)

        if input.is_cuda:
            mask = torch.from_numpy(mask).cuda(input.get_device())
        else:
            mask = torch.from_numpy(mask).to(self.device)

        input = input.permute(0, 2, 1)  # input: [N, T, in_channels]
        keys = self.linear_keys(input)  # keys: (N, T, key_size)
        query = self.linear_query(input)  # query: (N, T, key_size)
        values = self.linear_values(input)  # values: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2))  # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))

        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1)
        value_attentioned = torch.bmm(weight_temp, values).permute(0, 2, 1)  # shape: (N, value_size, T)

        return value_attentioned, weight_temp  # value_attentioned: [N, value_size, T], weight_temp: [N, T, T]


class CASTMGCN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, copy_supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, fusion='sum', supports_list=[6, 6, 6, 4]):
        super(CASTMGCN, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.supports_list = supports_list
        self.filter_convs = nn.ModuleList()
        self.filter_convs_reverse = nn.ModuleList()

        self.gate_convs = nn.ModuleList()
        self.gate_convs_reverse = nn.ModuleList()

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        # print('in_dim :{}，residual_channels:{}'.format(in_dim, residual_channels))

        self.supports = copy_supports
        self.device = device

        receptive_field = 1

        if gcn_bool and addaptadj:
            if aptinit is None:
                if copy_supports is None:  # If empty
                    self.supports = []  # self.supports is empty
                print('this')
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 32).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(32, num_nodes).to(device), requires_grad=True).to(device)
            else:
                m, p, n = torch.svd(aptinit.squeeze())
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                # print('(1, kernel_size)', (1, kernel_size))
                # print('new_dilation',new_dilation)
                self.filter_convs_reverse.append(nn.Conv2d(in_channels=residual_channels,
                                                           out_channels=dilation_channels,
                                                           kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_reverse.append(nn.Conv2d(in_channels=residual_channels,
                                                         out_channels=dilation_channels,
                                                         kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(MultiGCN(dilation_channels, residual_channels, dropout,
                                               support_list=self.supports_list, fusion=fusion))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

        self.atten = AttentionBlock(in_channels=13, key_size=dilation_channels, value_size=13, device=device)
        self.inhin = dilation_channels

    def forward(self, input, input_supports):
        B, N = input.shape[0], input.shape[2]

        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)

        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        # supports = supports.append()

        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adp = adp.unsqueeze(0)

            if adp is not None:
                input_supports = input_supports + [adp]
            else:
                input_supports = input_supports

                # print('x shape:', x.shape)
        x = x.permute(0, 2, 3, 1).reshape(-1, 13, self.inhin)
        value_attentioned, _ = self.atten(x)

        x = value_attentioned.reshape(B, N, 13, self.inhin).permute(0, 3, 1, 2)
        # WaveNet layers

        for i in range(self.blocks * self.layers):

            residual = x

            residual_reverse = residual.flip(-1)

            filter = self.filter_convs[i](residual)
            filter_reverse = self.filter_convs_reverse[i](residual_reverse)
            filter = filter + filter_reverse
            filter = torch.tanh(filter)

            gate = self.gate_convs[i](residual)
            gate_reverse = self.gate_convs_reverse[i](residual_reverse)
            gate = gate + gate_reverse
            # gate = gate
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if self.gcn_bool and input_supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, input_supports)
                else:
                    x = self.gconv[i](x, input_supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

    def get_max_selection_stats(self):
        """Obtain the max selection statistics of all MultiGCN layers"""
        if not self.gcn_bool:
            return "The model does not use GCN"

        all_stats = []
        for i, gconv in enumerate(self.gconv):
            stats = gconv.get_max_selection_stats()
            if isinstance(stats, dict):
                all_stats.append({
                    "layer": i + 1,
                    "stats": stats
                })

        if not all_stats:
            return "The max fusion method was not used"

        return all_stats
