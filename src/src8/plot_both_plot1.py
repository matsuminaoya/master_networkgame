import os
import time
import random
import inspect
import numpy as np #pip install numpy
import pandas as pd #pip install pandas
import networkx as nx #pip install networkx
import seaborn as sns #pip install seaborn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from statistics import mean

names = ["7_both_null_noreset_t10_g10000_r100_w5000_b1",] #TODO:

def Graph_plot_tc_tl_tf(csv, name, generation):
    data =pd.read_csv(csv)
    data_step = data[data["Generation"] == generation]

    df = data_step
    plt.figure(figsize=(12,10))
    plt.scatter(df["tl"],df["tc"],color="black",alpha=0.7)
    plt.xlabel("tl")
    plt.xlim(0.0, 1.1)
    plt.ylabel("tc")
    plt.ylim(0.0, 1.1)
    plt.title(f'generation {generation}')

    plt.savefig("pic8/"+name+"_plot1_tctl_g"+str(generation)+".png") #TODO:
    plt.close()

for name in names:
    Graph_plot_tc_tl_tf("D:/master/src8/"+name+".csv", name=name, generation=10000) #TODO: