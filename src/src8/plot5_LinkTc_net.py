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

names = ["7_anor_both_null_noreset_t10_g10000_r100_w5000_b1",
         ] #TODO:

def Graph_plot_tc_tl_tf(csv, name, generation):
    data =pd.read_csv(csv)
    data_step = data[data["Generation"] == generation]
    # tcとtlの組み合わせごとの件数をカウント（ピボットテーブル形式）
    heatmap_data = data_step.groupby(["tl","tf"]).size().unstack(fill_value=0)

    # 軸のラベル（順序）を設定
    tf_labels = np.round(np.linspace(0.0, 1.1, num=12), 1)
    tl_labels = np.round(np.linspace(1.1, 0.0, num=12), 1)

    # ラベルを明示的に並べ替え（不足分は自動で追加され0になる）
    heatmap_data = heatmap_data.reindex(index=tl_labels, columns=tf_labels, fill_value=0)

    plt.figure(figsize=(12,10))
    sns.heatmap(heatmap_data, cmap="tab20", annot=True, fmt='d', cbar=True, vmin=0, vmax=1000)
    plt.ylabel("tl")
    plt.xlabel("tf")
    plt.title(f'generation {generation}')

    plt.savefig("pic81/"+name+"_plot4_tltf_g"+str(generation)+".png") #TODO:
    plt.close()

for name in names:
    Graph_plot_tc_tl_tf("D:/master/src8/"+name+".csv", name=name, generation=10000) #TODO: