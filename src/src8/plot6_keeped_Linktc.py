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

#keeped or-leave
names = [#"3_keepd_01_leave_or_t10_g10000_r100_w1000_b1",
        #  "3_keepd_01_leave_or_t10_g10000_r100_w3000_b1",
        #  "3_keepd_01_leave_or_t10_g10000_r100_w5000_b1",
        #  "3_keepd_01_leave_or_t10_g10000_r100_w10000_b1",

        #  "3_keepd_03_leave_or_t10_g10000_r100_w1000_b1",
        #  "3_keepd_03_leave_or_t10_g10000_r100_w3000_b1",
        #  "3_keepd_03_leave_or_t10_g10000_r100_w5000_b1",
        #  "3_keepd_03_leave_or_t10_g10000_r100_w10000_b1",

        #  "3_keepd_05_leave_or_t10_g10000_r100_w1000_b1",
        #  "3_keepd_05_leave_or_t10_g10000_r100_w3000_b1",
        #  "3_keepd_05_leave_or_t10_g10000_r100_w5000_b1",
        #  "3_keepd_05_leave_or_t10_g10000_r100_w10000_b1",

         "3_keepd_01_form_an_t10_g10000_r100_w1000_b1",
         "3_keepd_01_form_an_t10_g10000_r100_w3000_b1",
         "3_keepd_01_form_an_t10_g10000_r100_w5000_b1",
         "3_keepd_01_form_an_t10_g10000_r100_w10000_b1",
         
         "3_keepd_03_form_an_t10_g10000_r100_w1000_b1",
         "3_keepd_03_form_an_t10_g10000_r100_w3000_b1",
         "3_keepd_03_form_an_t10_g10000_r100_w5000_b1",
         "3_keepd_03_form_an_t10_g10000_r100_w10000_b1",

        #  "3_keepd_05_form_an_t10_g10000_r100_w1000_b1",
        #  "3_keepd_05_form_an_t10_g10000_r100_w3000_b1",
        #  "3_keepd_05_form_an_t10_g10000_r100_w5000_b1",
        #  "3_keepd_05_form_an_t10_g10000_r100_w10000_b1",
         ] #TODO:

def Graph_plot_tc_link(csv, name, generation):
    data =pd.read_csv(csv)
    data_step = data[data["Generation"] == generation]
    # tcとtlの組み合わせごとの件数をカウント（ピボットテーブル形式）
    count_data = (
        data_step.groupby(["tf", "link_count"])
        .size()
        .reset_index(name="count")
    )

    plt.figure(figsize=(6,5))

    plt.scatter(
        count_data["link_count"],
        count_data["tf"],
        s=count_data["count"]*5,   # サイズで件数表現
        alpha=0.6
    )

    plt.xlabel("link")
    plt.ylabel("tf")
    #plt.title("",fontsize=20)

    plt.xlim(-5, 105)      # link を 0〜99
    plt.ylim(-0.05, 1.15)   # tc を 0.0〜1.1

    plt.xticks(range(0, 101, 10))
    plt.yticks(np.arange(0.0, 1.2, 0.1))

    plt.savefig("pic83/"+name+"_plot6_tflink_"+str(generation)+".png") #TODO:
    plt.close()

for name in names:
    Graph_plot_tc_link("D:/master/src8/"+name+".csv", name=name, generation=10000) #TODO: