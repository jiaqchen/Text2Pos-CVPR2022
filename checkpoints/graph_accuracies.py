# Open all json files that end with "*_retrieval_accuracies_pretrained_True.json" and plot the accuracies

# Open all json files that end with "*_retrieval_accuracies.json" and plot the accuracies

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

def plot():
    # list files in current directory
    list_dir = os.listdir('./')
    print(list_dir)

plot()