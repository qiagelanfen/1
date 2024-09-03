import torch
import torch.nn as nn
from utils.pinyu import pinyu
from utils.LMST import get_features

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.gcn_conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.gcn_conv(x, edge_index)


class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()
        self.means = None
        self.stds = None

    def norm(self, x):
        self.means = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - self.means
        self.stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / self.stds
        return x

    def denorm(self, x):
        x = x * self.stds + self.means
        return x


class EvidenceMachineKernel(nn.Module):
    def __init__(self, C, F):
        super(EvidenceMachineKernel, self).__init__()
        self.C = C
        self.F = 2 ** F
        self.C_weight = nn.Parameter(torch.randn(self.C, self.F))
        self.C_bias = nn.Parameter(torch.randn(self.C, self.F))

    def forward(self, x):
        x = torch.einsum('btc,cf->btcf', x, self.C_weight) + self.C_bias
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.piny = pinyu(self.configs.c_out).to(self.device)
        self.xianxing1=nn.Linear(self.piny.shape[1],configs.c_out)
        self.xianxing2 = nn.Linear(self.piny.shape[0], configs.batch_size*(self.pred_len + self.seq_len))
        #self.lm = get_features(32, self.pred_len + self.seq_len, self.configs.c_out).to(self.device)
        if self.task_name.startswith('long_term_forecast') or self.task_name == 'short_term_forecast':
            self.nl = NormLayer()
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.T_model = EvidenceMachineKernel(self.pred_len + self.seq_len, self.configs.e_layers)
    def z_score_normalize(self,matrix):
        mean = matrix.mean()
        std = matrix.std()
        normalized_matrix = (matrix - mean) / std
        return normalized_matrix

    def add(self,a,b):
        return a+b


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc [B, T, C]
        b=x_enc.shape[0]
        c=x_enc.shape[2]
        x = self.nl.norm(x_enc)
        # x [B, T, C]
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        pi = self.xianxing1(self.piny)
        pi = self.xianxing2(pi.permute(1, 0))
        pi = pi.permute(1, 0)
        pi = pi.reshape(b, self.pred_len + self.seq_len, c, 1)
        #lm=self.lm
        gate = torch.sigmoid(self.add(self.T_model(x.permute(0, 2, 1)).permute(0, 2, 1, 3), self.z_score_normalize(pi)))
        fused_features = gate * self.T_model(x.permute(0, 2, 1)).permute(0, 2, 1, 3) + (1 - gate) * self.z_score_normalize(pi)

        x = fused_features
        #x = self.T_model(x.permute(0, 2, 1)).permute(0, 2, 1, 3)
        x = torch.einsum('btcf->btc', x)
        x = self.nl.denorm(x)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name.startswith('long_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
