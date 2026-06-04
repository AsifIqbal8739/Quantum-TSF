# From here, we will be calling multiple functions to perform the model training and testing.
# Main variable changes are going to be supplied from here, the basic ones will be set by the called files
import numpy as np
import matplotlib.pyplot as plt
import os
from importlib import reload
from Results_Gen.Helper_Utilities.config_parameters import setup_parameters
from Results_Gen.Helper_Utilities.exp_main import Exp_Main
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Call the file to generate the configuration variable
    config = setup_parameters()
    config.useQTM = True    # will be used in effective dimension estimation as their we need Estimator QNN

    # Trials would be over Models, circuit depth, trials, predictionLength
    model_types = ['Linear', 'QuReUp', 'QuTNN', 'RealAmp', 'EFSU2', 'Pauli2D', 'nLoc'] # Model names
    # model_types = ['nLoc'] # Model names

    maxTrials = 2 # (0, 1)
    maxDepth = 4
    predLengths = [1, 6, 8, 12, 18, 24, 30, 36, 48]
    # predLengths = [1, 8]
    test = True    # whether we should perform training or directly do testing using the saved model weights

    for mt, name in enumerate(model_types):
        config.model_type = name
        for cdepth in range(maxDepth):
            config.cct_depth = cdepth+1
            # if cdepth < 3: # to just train using circuit depth of 4
            #     continue
            if cdepth > 0 and name == 'Linear':
                continue
            if cdepth > 0 and name == 'QuTNN':      # depth doesnt matter for QuTNN
                continue
            for nT in range(maxTrials):
                exp_main = Exp_Main(config, trial=nT)
                print('==========================================================')
                print(f'\tTraining {~test} for Model: {name}, Depth: {cdepth + 1}, Trial: {nT + 1} ')
                print('==========================================================')
                if not test:
                    # train the model
                    _, train_loss = exp_main.train()
                for pl, length in enumerate(predLengths):
                    if length == 1:
                        metrics = exp_main.test(test=test)
                    else:
                        metrics = exp_main.pred(test=test, pred_length=length)

