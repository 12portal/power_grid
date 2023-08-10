import copy
import torch
import numpy as np
import pandas as pd


def get_D_AE_data(path):
    def get_tensor_from_pd(dataframe_series) -> torch.Tensor:
        return torch.nn.functional.normalize(torch.tensor(data=dataframe_series.values), dim=0)

    data_x = pd.read_csv(path)  #'data/data_full.csv'
    data_x.columns = ["x{}".format(i + 1) for i in range(15)] + ["miss_line"]
    true_data = copy.deepcopy(data_x)
    # 在miss_line这一列删除数据,来模拟缺失值的场景
    drop_index = data_x.sample(frac=0.1).index  # 有缺失值的index
    data_x.loc[drop_index, "miss_line"] = np.nan
    true_value = true_data.loc[drop_index, 'miss_line']  # 空值的真实值
    # 开始构造数据
    # data_x为全部的数据（包含完整数据、有缺失项的数据）
    full_x = data_x.drop(drop_index)
    lack_x = data_x.loc[drop_index]
    return get_tensor_from_pd(full_x).float(), get_tensor_from_pd(lack_x).float(), get_tensor_from_pd(true_value).float()


def get_L_MLP_data(path):
    data = pd.read_csv(path)  # 'data/data_full.csv'
    # print(data_x)
    data = np.array(data, dtype='float64')
    train_data = data[:int(data.shape[0] * 0.8), :]
    test_data = data[int(data.shape[0] * 0.8):, :]
    train_data = torch.nn.functional.normalize(torch.tensor(train_data), dim=0).float()
    test_data = torch.nn.functional.normalize(torch.tensor(test_data), dim=0).float()
    return train_data, test_data


def get_L_LSTM_data(path, timestep):
    data = pd.read_csv(path)  # 'data/data_full.csv'
    # print(data_x)
    data = np.array(data, dtype='float64')
    train_data = data[:int(data.shape[0] * 0.8), :]
    test_data = data[int(data.shape[0] * 0.8):, :]

    def create_dataset(data, n_predictions):
        data_X, data_Y = [], []
        for i in range(data.shape[0] - n_predictions):
            a = data[i:(i + n_predictions), :]
            data_X.append(a)
            b = data[i + n_predictions, -1]
            data_Y.append(b)
        data_X = np.array(data_X, dtype='float64')
        data_Y = np.array(data_Y, dtype='float64')

        return data_X, data_Y

    train_x, train_y = create_dataset(train_data, timestep)
    test_x, test_y = create_dataset(test_data, timestep)
    # print(x.shape)
    # print(y.shape)

    train_x = torch.nn.functional.normalize(torch.tensor(train_x), dim=0).float()
    train_y = torch.nn.functional.normalize(torch.tensor(train_y), dim=0).float()
    test_x = torch.nn.functional.normalize(torch.tensor(test_x), dim=0).float()
    test_y = torch.nn.functional.normalize(torch.tensor(test_y), dim=0).float()
    return train_x, train_y, test_x, test_y