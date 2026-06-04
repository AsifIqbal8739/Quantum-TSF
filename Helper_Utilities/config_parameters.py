# DO NOT CHANGE!!!!! or be careful as it is used in multiple scripts to generate a baseline configuration
import argparse


def setup_parameters():
    # Configuration setting
    parameters = {}

    # basic config:
    # parameters['root_path'] = 'F:\\NUS Dropbox\\Asif Iqbal\\NUS Research\\Code\\QuantumML\\TSF-QML'
    parameters['root_path'] = '/Users/asif/NUS/NUS Research/Code/QuantumML/TSF-QML'
    parameters['data_path'] = 'Dataset/electricity_cl_2.csv'
    parameters['results_folder'] = 'Results_Gen/Results'
    parameters['num_client'] = 44  # which client TS to get or None for all
    parameters['show_client'] = False
    parameters['use_gpu'] = True

    # forecasting:
    parameters['seq_len'] = 12
    parameters['pred_len'] = 1
    parameters['maxlen'] = 24 * 15  # or None for entire TS
    parameters['trteperc'] = [0.6, 0.3]  # Train - Test Split

    # Common Model Training Parameters
    parameters['train_epochs'] = 150
    parameters['batch_size'] = 256
    parameters['learning_rate'] = [2*10**-3, 10**-2, 10**-3]    # always keep it as a list
    parameters['train_only'] = True
    parameters['swlr'] = [20, 120]    # first and second change of learning rate (Always keep it as a list)

    # Linear Model Parameters
    parameters['model'] = 'Linear'            # Select model here
    parameters['num_nodes'] = [48]              # number of nodes in the mid layer
    parameters['activation'] = 'GELU'         # GELU, ReLU, LeakyReLU

    # Below we setup different qunatum circuit parameters
    parameters['num_threads'] = 3     # for quantum torch module

    # QuReUp Model
    parameters['QuReUp_type'] = 2     # 1-> Data Reuploading at layer end, 2-> at beginning
    parameters['QuReUp_Ent'] = 'full' # full or nn  === this change got the results to bingo

    # QuTNN Model
    parameters['ctype'] = 'top'           # ctype = 'bot' or 'top': Bottom or Top Qubit to be measured
    parameters['rottype'] = 'rr'        # rotation types for the 2 single bit rotations or R(theta, phi) | 'yz' or 'rr'
    parameters['input_bias'] = False       # bias included in the encoding or not
    parameters['enttype'] = 'full'        # nn -> cx or full -> cz ent type

    # BuiltIn Models: # RealAmp, EFSU2, Pauli2D, nLoc
    parameters['entang'] = 'full'     # full, linear, reverse_linear, circular

    config = argparse.Namespace(**parameters)
    return config