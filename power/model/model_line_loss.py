import torch.nn as nn
import math


def local_method(data):
    #  基础方法
    data[u'线损率'] = (data[u'供入电量'] - data[u'供出电量']) / data[u'供入电量']
    return data[u'线损率']


def equivalent_resistance_method(data):
    #  等效电阻法
    N = 2  # N为电网结构系数，当为单相供电时，N = 2；当为三相三线制时，N = 3；当为三相四线制时，N = 3.5
    data[u'线损率'] = N * pow((data[u'形状因子'] * data[u'电力线头部的平均电流']), 2) * data[u'低压电力线的等效电阻'] * \
                      data[u'操作时间'] * pow(10, -3)
    return data[u'线损率']


def voltage_loss_method(data):
    #  电压损失法
    u = (data[u'供入电压'] - data[u'供出电压']) / data[u'供入电压']
    k = (1 + pow(math.tan(data[u'电流比电压滞后角度']), 2)) / (1 + data[u'导线X/R值'] * math.tan(data[u'电流比电压滞后角度']))
    data[u'线损率'] = k * u
    return data[u'线损率']


# MLP预测
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.linear1(input)
        input = self.sigmoid(input)
        input = self.linear2(input)
        input = self.sigmoid(input)
        input = self.linear3(input)
        return input


# LSTM预测
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq) # output(5, 30, 64)
        pred = self.linear(output)  # (5, 30, 1)
        pred = pred[:, -1, :]  # (5, 1)
        return pred
