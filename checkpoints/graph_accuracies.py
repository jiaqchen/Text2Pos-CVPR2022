# Open all json files that end with "*_retrieval_accuracies_pretrained_True.json" and plot the accuracies

# Open all json files that end with "*_retrieval_accuracies.json" and plot the accuracies

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

def get_accuracies_scratch(list_dir):
    epochs = []
    mean_accuracies = {}
    variances = {}
    for file in list_dir:
        with open(file, 'r') as f:
            epoch_number = int(file.split('_')[5])
            data = f.read().replace("(", "[").replace(")", "]")
            data = data.split('_')
            data = eval(data[1].strip())
            epochs.append(epoch_number)
            for k in data:
                if k not in mean_accuracies:
                    mean_accuracies[k] = []
                mean_accuracies[k].append(data[k][0])
                if k not in variances:
                    variances[k] = []
                variances[k].append(data[k][1])

    print(f'Epochs: {epochs}')
    print(f'Mean Accuracies: {mean_accuracies}')
    print(f'Variances: {variances}')
    return epochs, mean_accuracies, variances

def get_accuracies_pretrain(list_dir):
    # sort list_dir by the number
    list_dir = sorted(list_dir, key=lambda x: int(re.findall(r'\d+', x)[0]))
    epochs = []
    mean_accuracies = {}
    variances = {}
    for file in list_dir:
        with open(file, 'r') as f:
            epoch_number = int(file.split('_')[5])
            data = f.read().replace("(", "[").replace(")", "]")
            data = data.split('_')
            data = eval(data[1].strip())
            epochs.append(epoch_number)
            for k in data:
                if k not in mean_accuracies:
                    mean_accuracies[k] = []
                mean_accuracies[k].append(data[k])
                # if k not in variances:
                #     variances[k] = []
                # variances[k].append(data[k][1])

    print(f'Epochs: {epochs}')
    print(f'Mean Accuracies: {mean_accuracies}')
    # print(f'Variances: {variances}')
    return epochs, mean_accuracies, None

def plot():
    list_dir = os.listdir('./')

    list_dir_scratch = sorted([x for x in list_dir if x.endswith('False.json') and 'args' not in x])
    list_dir_pretrain = sorted([x for x in list_dir if x.endswith('accuracies.json') and 'args' not in x])

    # Open and get accruacy values
    scratch_epochs, scratch_mean_acc, scratch_var = get_accuracies_scratch(list_dir_scratch) # epoch: top k accuracies
    pretrain_epochs, pretrain_mean_acc, pretrain_var = get_accuracies_pretrain(list_dir_pretrain) # epoch: top k accuracies

    # Plot the accuracies
    plt.figure()
    for k in scratch_mean_acc:
        plt.errorbar(scratch_epochs, scratch_mean_acc[k], label=f'Scratch {k}')
    for k in pretrain_mean_acc:
        plt.errorbar(pretrain_epochs, pretrain_mean_acc[k], label=f'Pretrained {k}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.show()

plot()