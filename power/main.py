import torch
from power.model.train_power import D_AE_train, L_MLP_train, L_LSTM_train


if __name__ == "__main__":
    task = 'D_AE'                           # 'D_AE', 'D_MedianFilter', 'D_MeanFilter', 'L_MLP', 'L_LSTM'
    if task == 'D_AE':
        print('开始数据修补，方法为AutoEncoder')
        path = 'data/data_full.csv'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epochs = 2

        D_AE_train(path=path, device=device, epochs=epochs)

    if task == 'L_MLP':
        print('开始线损率计算，方法为MLP')
        path = 'data/data_full.csv'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        epochs = 2

        L_MLP_train(path=path, device=device, epochs=epochs)

    if task == 'L_LSTM':
        print('开始线损率计算，方法为LSTM')
        path = 'data/data_full.csv'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        timestep = 10
        epochs = 2

        L_LSTM_train(path=path, device=device, timestep=timestep, epochs=epochs)







