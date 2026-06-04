import numpy as np
import matplotlib.pyplot as plt

# plt.ion()
import os


def adjust_lr(optimizer, epoch, config):
    # if isinstance(config.swlr, list): # when config.learning_rate is a list of lrs [start, mid, end]
    if len(config.swlr) > 1:
        if epoch == config.swlr[0]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate[1]
            print('Updating the learning rate to {}'.format(config.learning_rate[1]))
        elif epoch == config.swlr[1]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate[2]
            print('Updating the learning rate to {}'.format(config.learning_rate[2]))
    elif epoch == config.swlr[0]:  # when swlr is a single number
        for param_group in optimizer.param_groups:
            lr = param_group['lr'] / 10
            param_group['lr'] = lr
        print('Updating the learning rate to {}'.format(lr))


# code to read the saved metrics scores and make them presentable
def ext_metrics(config, max_trails=1, pred_lengths=[1], verbose=False):
    from Results_Gen.Helper_Utilities.exp_main import Exp_Main

    name = config.model_type
    for pl, length in enumerate(pred_lengths):
        for nT in range(max_trails):
            exp_main = Exp_Main(config, trial=nT)
            file_path = os.path.join(exp_main.folder_path, f'{exp_main.file_name}_pred_{length}-metrics.npy')
            if verbose:
                print(file_path[-45:])
            temp = np.load(file_path)
            metrics = temp if nT == 0 else np.vstack((metrics, temp))
        avg = np.average(metrics, axis=0)
        metrics_pl = avg if pl == 0 else np.vstack((metrics_pl, avg))

    return metrics_pl


# code for displaying the predictions
def disp_predicts(config, nTr=1, predL=1, startInd=2):
    # startInd can start from zero,
    folder_path = os.path.join(config.root_path, config.results_folder, config.model_type,
                               f'cctDepth_{config.cct_depth}')
    file_name = f'trial_{nTr}'
    name = config.model_type
    seq_len = config.seq_len

    preds = np.load(os.path.join(folder_path, f'{file_name}_pred_{predL}-pred.npy'))
    trues = np.load(os.path.join(folder_path, f'{file_name}_pred_{predL}-trues.npy'))

    if predL != 1:
        plt.figure(figsize=[8, 4], dpi=100)
        vectP = preds[:, startInd]
        vectT = np.squeeze(trues[:, startInd:startInd + predL])

        plt.plot(vectP, '--*', label='Predictions')
        plt.plot(vectT, label='Trues')
    else:
        plt.figure(figsize=[8, 4], dpi=100)
        plt.plot(preds, '--*', label='Predictions')
        plt.plot(trues, label='Trues')
    plt.xlabel('Time Index')
    # plt.title(f'Prediction length {predL}, starting at {startInd}, by Model: {name}')
    plt.title(f'Prediction length {predL} by Model: {name}, Depth: {config.cct_depth}')

    plt.legend()
    plt.grid()
    plt.show()


# ============= Generating plots for multi-sample predictions using MultiTrial_2 checkpoints
def disp_pred_win(config, nTr=1, predL=1, startInd=2, mt=0, combined_plot=False, true_plot=True):
    # startInd can start from zero, combined plots to be used or individual
    # mt is needed when the plots are on the same figure
    folder_path = os.path.join(config.root_path, config.results_folder, config.model_type,
                               f'cctDepth_{config.cct_depth}')
    file_name = f'trial_{nTr}'
    name = config.model_type
    seq_len = config.seq_len
    print(f'Loading data from folder path: {folder_path}')
    preds = np.load(os.path.join(folder_path, f'{file_name}_pred_{predL}-pred.npy'))
    trues = np.load(os.path.join(folder_path, f'{file_name}_pred_{predL}-trues.npy'))

    # Generating a single time series from batched windows
    [nB, nP] = preds.shape
    preds_cum, trues_cum = np.zeros((nB+nP,)), np.zeros((nB+nP,))
    # as for recursive results, the trues are a single vector, we need to be careful
    if predL != 1:
        trues_cum = trues
    for nn in range(nB):
        preds_cum[nn:nn+nP] += preds[nn, :]
        if predL == 1:
            trues_cum[nn:nn + nP] += trues[nn, :]

    nP = np.min((nB, nP))    # for plots in the same figure
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '*', 'x', '+']
    colors = plt.cm.tab10.colors

    if combined_plot is False:
        plt.figure(figsize=[8, 4], dpi=100)
        plt.plot(preds_cum[startInd:startInd+48]/nP, '--*', label='Predictions')
        if predL == 1:
            plt.plot(trues_cum[startInd:startInd+48]/nP, label='Trues')
        else:
            plt.plot(trues_cum[0, startInd:startInd+48], label='Trues')

        plt.xlabel('Time Index')
        # plt.title(f'Prediction length {predL}, starting at {startInd}, by Model: {name}')
        plt.title(f'Native Predictions by Model: {name}, Depth: {config.cct_depth}')
    else:
        if true_plot:
            if predL == 1:
                plt.plot(trues_cum[startInd:startInd + 48] / nP, label='Trues', marker='x',
                     color='k', markevery=20, linewidth=2)
            else:
                plt.plot(trues_cum[0,startInd:startInd + 48], label='Trues', marker='x',
                     color='k', markevery=20, linewidth=2)

        plt.plot(preds_cum[startInd:startInd+48]/nP, label=f'{name}',
             linestyle=line_styles[mt % len(line_styles)],
             marker=markers[mt % len(markers)],
             color=colors[mt % len(colors)],
             markevery=4)

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def disp_pred_block(config, nTr=1, predL=1, startInd=2, combined_plot=False, true_plot=True):
    # startInd can start from zero, combined plots to be used or individual
    # plotting block wise, not averaging over predicitons
    folder_path = os.path.join(config.root_path, config.results_folder, config.model_type,
                               f'cctDepth_{config.cct_depth}')
    file_name = f'trial_{nTr}'
    name = config.model_type
    seq_len = config.seq_len
    print(f'Loading data from folder path: {folder_path}')
    preds = np.load(os.path.join(folder_path, f'{file_name}_pred_1-pred.npy'))
    trues = np.load(os.path.join(folder_path, f'{file_name}_pred_1-trues.npy'))

    [nB, nP] = preds.shape
    pred_f = np.reshape(preds[::nP, :], -1)
    true_f = np.reshape(trues[::nP, :], -1)

    if combined_plot is False:
        plt.figure(figsize=[8, 4], dpi=100)
        plt.plot(pred_f[startInd:startInd+48], '--*', label='Predictions')
        plt.plot(true_f[startInd:startInd+48], label='Trues')
        plt.xlabel('Time Index')
        # plt.title(f'Prediction length {predL}, starting at {startInd}, by Model: {name}')
        plt.title(f'Native Predictions by Model: {name}, Depth: {config.cct_depth}')
    else:
        if true_plot:
            plt.plot(true_f[startInd:startInd + 48], label='Trues')
        plt.plot(pred_f[startInd:startInd+48], label=f'{name}')

    plt.legend()
    plt.grid()
    plt.show()

def disp_wins(config, nTr=0, numb=None, random=False):
    # numb for the specific batch sample to plot, random: plot some random samples
    folder_path = os.path.join(config.root_path, config.results_folder, config.model_type,
                               f'cctDepth_{config.cct_depth}')
    file_name = f'trial_{nTr}'
    name = config.model_type
    seq_len = config.seq_len
    preds = np.load(os.path.join(folder_path, f'{file_name}_pred_1-pred.npy'))
    trues = np.load(os.path.join(folder_path, f'{file_name}_pred_1-trues.npy'))

    if random is True:
        indx = np.random.permutation(preds.shape[0])[:6]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # 2 rows, 3 columns

        for i, ax in enumerate(axes.flat):  # Flatten the 2D array of axes
            mae = np.mean(np.abs(preds[indx[i], :].reshape(-1, 1) - trues[indx[i], :].reshape(-1, 1)))
            mse = np.mean(np.square(preds[indx[i], :].reshape(-1, 1) - trues[indx[i], :].reshape(-1, 1)))
            ax.plot(preds[indx[i], :].reshape(-1, 1), '--^', label='Predictions')  # Example plots with different scales
            ax.plot(trues[indx[i], :].reshape(-1, 1), label='Trues')
            ax.set_title(f"Index: {indx[i]}, MAE: {mae:0.2f}, MSE: {mse:0.2f}")
            ax.set_ylim(-1.0, 1.0)

        plt.tight_layout(pad=1.0)  # Adjust spacing between subplots
        plt.legend()
        plt.show()
        return
