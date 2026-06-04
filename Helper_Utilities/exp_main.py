# Main file to setup the experiment
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time

from Results_Gen.Helper_Utilities.data_factory import data_provider
from Results_Gen.Helper_Utilities.metrics import metric
import Results_Gen.Helper_Utilities.utilities as ut
from Results_Gen.Helper_Utilities.model_factory import ModelFactory


class Exp_Main(object):
    def __init__(self, config, trial=0):
        self.config = config
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)    # Creating of model at object creation
        self.trial = trial      # Current trial number for proper results storage

        self.folder_path = os.path.join(self.config.root_path, self.config.results_folder, self.config.model_type, f'cctDepth_{self.config.cct_depth}')
        self.file_name = f'trial_{self.trial}'

    def _acquire_device(self):
        if self.config.use_gpu and self.config.model == 'Linear':
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            print('Use GPU: {}'.format(device))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = ModelFactory(config=self.config)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.config, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config.learning_rate[0])
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for nb, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        if not self.config.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        train_steps = len(train_loader)
        epoch_loss = []

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        folder_path = self.folder_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for epoch in range(self.config.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for nb, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.squeeze(-1)
                batch_y = batch_y.squeeze(-1)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.to(torch.float32).to(self.device)
                batch_y = batch_y.to(torch.float32).to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if nb % 2 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(nb + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.train_epochs - epoch) * train_steps - nb)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            epoch_loss.append(train_loss)

            if not self.config.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))

            ut.adjust_lr(model_optim, epoch+1, self.config)

        torch.save(self.model.state_dict(), os.path.join(folder_path, f'{self.file_name}-checkpoint.pth'))

        np.save(os.path.join(folder_path, f'{self.file_name}-epoch_loss.npy'), epoch_loss)
        return self.model, epoch_loss

    def test(self, test=True):
        est_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        inputx = []

        folder_path = self.folder_path
        if test:
            print('Loading Model....')
            self.model.load_state_dict(torch.load(os.path.join(folder_path, f'{self.file_name}-checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            for nb, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.to(torch.float32).to(self.device)
                batch_y = batch_y.to(torch.float32).to(self.device)

                batch_x = batch_x.squeeze(-1)
                batch_y = batch_y.squeeze(-1)

                outputs = self.model(batch_x)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                inputx.append(batch_x)

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            inputx = np.concatenate(inputx, axis=0)

            mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
            metrics = [mae, mse, rmse, mape, mspe, rse, corr]
            print('MAE: {}, MSE: {}, RMSE: {}, MAPE: {}, RSE: {}, CORR: {}'.format(mae, mse, rmse, mape, mspe, rse,
                                                                                   corr))
            np.save(os.path.join(folder_path, f'{self.file_name}_pred_1-pred.npy'), preds)
            np.save(os.path.join(folder_path, f'{self.file_name}_pred_1-trues.npy'), trues)
            np.save(os.path.join(folder_path, f'{self.file_name}_pred_1-metrics.npy'), metrics)

        return metrics


    def pred(self, test=False, pstart=0, pred_length=6):
        # test - True -> load checkpoint
        # pred - True -> perform iterative prediction
        # pstart - From which time point the iterative prediction to begin

        input_data, _ = self._get_data(flag='pred')    # convert to matrix of size [1 x totalsamples] -> batche
        input_size = len(input_data)
        input_data = input_data.T
        metrics = []
        folder_path = self.folder_path
        if test:
            print('Loading Model....')
            self.model.load_state_dict(torch.load(os.path.join(folder_path, f'{self.file_name}-checkpoint.pth')))

        self.model.eval()
        with torch.no_grad():
            for ii in range(input_size - self.config.seq_len - pred_length - 1):   # loop through the whole time series
                input_x = input_data[:, ii: ii+self.config.seq_len]
                input_x = torch.from_numpy(input_x).to(torch.float32).to(self.device)
                preds = []
                for jj in range(pred_length):
                    output_x = self.model(input_x)
                    temp = torch.cat((input_x[:, 1:], output_x), dim=1)
                    input_x = temp

                    outputs = output_x.detach().cpu().numpy().item()
                    preds.append(outputs)
                istart = ii+self.config.seq_len
                trues = np.expand_dims(input_data[0, istart:istart+pred_length], axis=1)
                preds = np.expand_dims(np.array(preds), axis=1)
                mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
                scores = [mae, mse, rmse, mape, mspe, rse, corr]
                metrics.append(scores)

                predicts = np.array(preds) if ii == 0 else np.hstack((predicts, np.array(preds)))
        metrics = np.mean(np.concatenate(metrics, axis=0).reshape((-1, len(scores))), axis=0)
        np.save(os.path.join(folder_path, f'{self.file_name}_pred_{pred_length}-pred.npy'), predicts)
        np.save(os.path.join(folder_path, f'{self.file_name}_pred_{pred_length}-trues.npy'), input_data[:, self.config.seq_len:])
        np.save(os.path.join(folder_path, f'{self.file_name}_pred_{pred_length}-metrics.npy'), metrics)

        return metrics