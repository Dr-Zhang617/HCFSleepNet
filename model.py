import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficient_kan import KAN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()

        # Squeeze操作
        self.squeeze = nn.AdaptiveAvgPool1d(1)

        # Excitation操作
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

        self.Residual = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(in_channels),
        )

    def forward(self, x):
        # 输入x的形状为 (batch_size, in_channels, feature_data)
        # Squeeze操作
        x = x
        x1 = self.Residual(x)
        y = self.squeeze(x1)
        # 将squeeze后的结果展平
        y = y.view(y.size(0), -1)
        # Excitation操作
        y = self.excitation(y)
        # 将得到的权重广播到每个通道
        y = y.view(y.size(0), -1, 1)
        # 对输入x进行加权
        out = x1 * y
        out = x + out
        return out

class Temporal_feature_EEG(nn.Module):
    def __init__(self, channels):
        super(Temporal_feature_EEG, self).__init__()
        drate = 0.5
        self.features1 = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=50, stride=8, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(128, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)

        )

        self.features2 = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=500, stride=64, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Conv1d(128, 64, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        # self.Flatten=nn.Flatten()
        self.dropout = nn.Dropout(drate)
        self.se_block = SEBlock(in_channels=64)

    def forward(self, y):
        if len(y.shape) == 2:
            y = y.unsqueeze(dim=1)
        y1 = self.features1(y)
        y2 = self.features2(y)
        y_concat = torch.cat((y1, y2), dim=2)
        y_concat = self.dropout(y_concat)
        y_concat = self.se_block(y_concat)
        return y_concat


class Temporal_feature_multimodel(nn.Module):
    def __init__(self, channels):
        super(Temporal_feature_multimodel, self).__init__()
        drate = 0.5
        self.features1 = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=25, stride=8, bias=False, padding=8),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 32, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)

        )

        self.features2 = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=250, stride=64, bias=False, padding=24),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(32, 64, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Conv1d(64, 32, kernel_size=6, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.se_block = SEBlock(in_channels=32)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.se_block(x_concat)
        return x_concat



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.W_q1 = nn.Linear(d_model, d_model, bias=False)
        self.W_q2 = nn.Linear(d_model, d_model, bias=False)
        self.KV = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(0.5)


    def forward(self, x1, x2, mask=None):
        # Linear transformation for each head
        Q11 = self.W_q1(self.layer_norm1(x1))
        Q22 = self.W_q2(self.layer_norm2(x2))
        FC = torch.cat((Q11, Q22), dim=1)
        kv = self.KV(FC)

        # Split the input into multiple heads
        Q1 = Q11.view(Q11.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        Q2 = Q22.view(Q22.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = kv.view(kv.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = kv.view(kv.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention
        scores1 = torch.matmul(Q1, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores2 = torch.matmul(Q2, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores1 = scores1.masked_fill(mask == 0, float('-inf'))

        if mask is not None:
            scores2 = scores2.masked_fill(mask == 0, float('-inf'))

        attention_weights1 = F.softmax(scores1, dim=-1)
        output1 = torch.matmul(attention_weights1, v)

        attention_weights2 = F.softmax(scores2, dim=-1)
        output2 = torch.matmul(attention_weights2, v)


        # Concatenate and linear transformation
        output1 = output1.transpose(1, 2).contiguous().view(output1.size(0), -1, self.num_heads * self.head_dim)
        output1 = output1 + Q11
        output2 = output2.transpose(1, 2).contiguous().view(output2.size(0), -1, self.num_heads * self.head_dim)
        output2 = output2 + Q22

        output1 = self.dropout(output1)
        output2 = self.dropout(output2)

        return output1, output2

class CrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(CrossModalAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.kan = KAN([d_model, 4 * d_model, d_model])

        self.layer_norm1 = nn.LayerNorm(d_model)

    def forward(self, x1, x2, mask=None):
        # Attention mechanism
        output1, output2 = self.attention(x1, x2, mask)

        output = torch.cat((output1, output2), dim=1)
        b, c, h = output.shape
        # Feedforward layer
        output = output + self.kan(self.layer_norm1(output).reshape(-1, output.shape[-1])).reshape(b, c, h)

        return output




class MMASleepNet(nn.Module):
    def __init__(self):
        super(MMASleepNet, self).__init__()

        self.d_model = 24
        self.nhead = 6
        self.EEG_channels = 2

        self.EEG_feature = Temporal_feature_EEG(channels=self.EEG_channels)
        self.EOG_feature = Temporal_feature_multimodel(channels=1)
        self.EMG_feature = Temporal_feature_multimodel(channels=1)

        self.cross_modal_attention = CrossModalAttention(self.d_model, self.nhead)

        self.kan = KAN([self.d_model*128, 32, 5])
        self.Softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x1, x2, x3 = torch.split(x, [2, 1, 1], dim=1)

        self.batch_size = x.shape[0]
        # channels = ['C4_A1', 'C3_A2', "ROC_A1", "X1"]
        x_EEG = self.EEG_feature(x1)
        x_EOG = self.EOG_feature(x2)
        x_EMG = self.EMG_feature(x3)
        out = self.cross_modal_attention(x_EEG, x_EOG)
        output = self.cross_modal_attention(out, x_EMG)
        output = output.view(self.batch_size, -1)
        output = self.kan(output)
        output = self.Softmax(output)
        return output
