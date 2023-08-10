import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from power.model.power_dataset import get_D_AE_data, get_L_LSTM_data, get_L_MLP_data
from power.model.model_data_processing import AutoEncoder
from sklearn.metrics import mean_squared_error
from power.model.model_line_loss import LSTM, MLP


def D_AE_train(path, device, epochs):
    full_data, lack_data, true_x = get_D_AE_data(path=path)
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(full_data),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=20,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=4,  # 多进程（multiprocess）来读数据
    )

    # 损失函数（MSE），优化器（Adam），epochs
    model = AutoEncoder(input_size=full_data.size()[1])  # 模型
    loss_function = nn.MSELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    # 开始训练
    model.to(device)
    model.train()
    for i in range(epochs):
        epoch_loss: list = []
        for seq in train_loader:
            seq = seq[0]
            seq = seq.to(device)
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            single_loss = loss_function(y_pred, seq)
            single_loss.backward()
            optimizer.step()
            epoch_loss.append(float(single_loss.detach()))
        print("EPOCH", i, "LOSS: ", np.mean(epoch_loss))
        # 每20次，输出一次前20个的结果，对比一下效果
    # 开始填充缺失值
    lack_loader = Data.DataLoader(
        dataset=Data.TensorDataset(lack_data),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=20,  # 每块的大小
        shuffle=True,  # 要不要打乱数据
        num_workers=4,  # 多进程（multiprocess）来读数据
    )
    model.eval()
    pred_lack = np.array([])
    for seq in lack_loader:
        seq = seq[0]
        seq = seq.to(device)
        # 每个seq[:,-1]都是缺失值的位置
        seq = torch.where(torch.isnan(seq), torch.full_like(seq, 0), seq)  # 全0填充缺失值
        y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
        lack_pred = y_pred[:, -1]  # AutoEncoder预测的缺失值
        pred_lack = np.append(pred_lack, np.array(lack_pred.detach().cpu().numpy()))

    print("预测结果的MSE：", mean_squared_error(true_x, pred_lack))


def L_MLP_train(path, device, epochs):
    full_data, test_data= get_L_MLP_data(path=path)
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(full_data),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=20,  # 每块的大小
        shuffle=True,  # 要不要打乱数据
        num_workers=4,  # 多进程（multiprocess）来读数据
    )
    model = MLP(input_size=full_data.size()[1])
    loss_function = nn.MSELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    # 开始训练
    model.to(device)
    model.train()
    for i in range(epochs):
        epoch_loss: list = []
        for seq in train_loader:
            seq = seq[0]
            seq = seq.to(device)
            optimizer.zero_grad()
            y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            single_loss = loss_function(y_pred, seq[:, -1])
            single_loss.backward()
            optimizer.step()
            epoch_loss.append(float(single_loss.detach()))
        print("EPOCH", i, "LOSS: ", np.mean(epoch_loss))
        # 每20次，输出一次前20个的结果，对比一下效果
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(test_data),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=20,  # 每块的大小
        shuffle=True,  # 要不要打乱数据
        num_workers=4,  # 多进程（multiprocess）来读数据
    )
    model.eval()
    pred_test = np.array([])
    for seq in test_loader:
        seq = seq[0]
        seq = seq.to(device)
        # 每个seq[:,-1]都是缺失值的位置
        seq = torch.where(torch.isnan(seq), torch.full_like(seq, 0), seq)  # 全0填充缺失值
        y_pred = model(seq).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
        test_pred = y_pred[:]  # AutoEncoder预测的缺失值
        pred_test = np.append(pred_test, np.array(test_pred.detach().cpu().numpy()))

    print("预测结果的MSE：", mean_squared_error(test_data[:, -1], pred_test))


def L_LSTM_train(path, device, timestep, epochs):
    train_x, train_y, test_x, test_y = get_L_LSTM_data(path=path, timestep=timestep)
    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(train_x),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=20,  # 每块的大小
        shuffle=False,  # 要不要打乱数据
        num_workers=0,  # 多进程（multiprocess）来读数据
    )

    # 损失函数（MSE），优化器（Adam），epochs
    model = LSTM(input_size=train_x.size()[2], hidden_size=64, num_layers=2, output_size=1, batch_size=True)  # 模型选择
    loss_function = nn.MSELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    # 开始训练
    model.to(device)
    model.train()
    for i in range(epochs):
        epoch_loss: list = []
        j = 0
        for seq in train_loader:
            seq = seq[0]
            datas = seq
            datas = datas.to(device)
            y = train_y.to(device)
            optimizer.zero_grad()
            y_pred = model(datas).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            if y_pred.shape[0] < 20:
                break
            single_loss = loss_function(y_pred, y[j:j + 20])
            single_loss.backward()
            optimizer.step()
            epoch_loss.append(float(single_loss.detach()))
            j = j + 1
        print("EPOCH", i, "LOSS: ", np.mean(epoch_loss))
        # 每20次，输出一次前20个的结果，对比一下效果

    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(test_x),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=20,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多进程（multiprocess）来读数据
    )
    model.eval()
    pred = []

    for seq in test_loader:
        seq = seq[0]
        datas = seq
        datas = datas.to(device)
        y = test_y.to(device)
        y_pred = model(datas).squeeze()  # 压缩维度：得到输出，并将维度为1的去除
        if y_pred.shape[0] < 20:
            break
        pred = np.append(pred, np.array(y_pred.detach().cpu().numpy()))

    print("预测结果的MSE：", mean_squared_error(test_y[:int(test_y.shape[0] - int(test_y.shape[0]) % 20)], pred))