from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# plt.ion()


class DatasetElectricity(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None, maxlen=None, trteperc=None, scale=True,
                 num_client=None):
        # size [seq_len, pred_len]
        if size is None:
            self.seq_len = 24
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        if trteperc is None:    # Pecentage of Train-Test set distribution
            self.train_perc = 0.7
            self.test_perc = 0.2
        else:
            self.train_perc = trteperc[0]
            self.test_perc = trteperc[1]

        self.root_path = root_path
        self.data_path = data_path
        self.num_client = num_client  # which client to choose or a list
        self.scale = scale
        # Which data to get
        assert flag in ['train', 'test', 'val', 'pred']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred': 2}
        self.set_type = type_map[flag]
        self.data_path = os.path.join(self.root_path, self.data_path)
        self.maxlen = maxlen    # the maximum length of the TS to load in

        self.__read_data__()

    def __read_data__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        df_raw = pd.read_csv(self.data_path)
        if self.maxlen is not None:
            df_raw = df_raw[:min(len(df_raw), self.maxlen)]

        cols = list(df_raw.columns)
        num_train = int(len(df_raw) * self.train_perc)
        num_test = int(len(df_raw) * self.test_perc)
        num_val = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - self.seq_len - num_test]
        border2s = [num_train, num_train + num_val, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.num_client is None:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[cols[self.num_client]]
            # plt.plot(df_data.values)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            if self.num_client is not None:
                self.scaler.fit(train_data.values.reshape(-1, 1))
                data = self.scaler.transform(df_data.values.reshape(-1, 1))
            else:
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)

        else:
            data = df_data.values

        self.data = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def data_provider(config, flag):
    train_only = config.train_only
    if flag == 'test':
        shuffle = False
        batch_size = config.batch_size
    elif flag == 'pred':    # legacy flag --- KEEP IT
        shuffle = False
        batch_size = 1
    else:
        shuffle = True
        batch_size = config.batch_size

    data_set = DatasetElectricity(root_path=config.root_path, data_path=config.data_path,
                                  flag=flag, size=[config.seq_len, config.pred_len], maxlen=config.maxlen,
                                  trteperc=config.trteperc, num_client=config.num_client)

    if config.show_client is True:
        plt.figure()
        plt.plot(data_set.data)
        plt.xlabel('Days - Index')
        plt.title('Scaled TS for {} data'.format(flag))

        # Set x-ticks to be every 24 hours (once per day)
        num_days = len(data_set.data) // 24  # How many full days exist in the data
        tick_positions = [i * 24 for i in range(num_days + 1) if i * 24 < len(data_set.data)]
        # Create labels for the days
        tick_labels = [f"Day {i}" for i in range(len(tick_positions))]
        plt.xticks(tick_positions, tick_labels, rotation=45)

        plt.show()

    if flag != 'pred':
        print(flag, len(data_set), 'samples')
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
        return data_set, data_loader
    else:
        return data_set.data, []