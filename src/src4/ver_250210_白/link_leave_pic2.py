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

names = ["t1_w5000_leave_full_nonono_20250211-0407",
         "t1_w5000_leave_full_noyesno_20250211-0407",
         "t1_w5000_leave_full_noyesyes_20250211-0407",
         "t1_w5000_leave_full_yesnono_20250211-0407",
         "t1_w5000_leave_full_yesyesno_20250211-0407",
         "t1_w5000_leave_full_yesyesyes_20250211-0407",]

def Graph_avr_tc_tl(csv, name):
    #allavr
    data =pd.read_csv(csv)
    avr_ges_all = data.groupby("ge")[["tc","tl","ln"]].mean().reset_index()
    
    df = avr_ges_all
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["ge"],df["tc"],label="tc",color="tab:blue")
    ax1.plot(df["ge"],df["tl"],label="tl",color="tab:orange")
    # ax1.plot(df["ge"],df["tf"],label="tf",color="tab:green")
    ax1.set_ylim(-0.09,1.19)
    ax1.set_xlabel("generation")
    ax2 = ax1.twinx()
    ax2.bar(df["ge"],df["ln"],color='lightblue',label="Link")
    ax2.set_ylim(-0.09,100.09)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc="upper right")
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    plt.savefig(name+"/"+name+"_allavr"+".png")
    plt.close()

    #linked
    data_linked = data[data["ln"] > 0]
    avr_ges_linked = data_linked.groupby("ge")[["tc","tl"]].mean().reset_index()

    df = avr_ges_linked
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["ge"],df["tc"],label="tc",color="tab:blue")
    ax1.plot(df["ge"],df["tl"],label="tl",color="tab:orange")
    # ax1.plot(df["ge"],df["tf"],label="tf",color="tab:green")
    ax1.set_ylim(-0.09,1.19)
    ax1.set_xlabel("generation")
    # ax2 = ax1.twinx()
    # ax2.bar(df["ge"],df["ln"],color='lightblue',label="Link")
    # ax2.set_ylim(0,100)
    h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1, l1 ,loc="upper right")
    ax1.set_zorder(1)
    # ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    plt.savefig(name+"/"+name+"_linked"+".png")
    plt.close()

    #nolink
    data_nolink = data[data["ln"] == 0]
    avr_ges_nolink = data_nolink.groupby("ge")[["tc","tl"]].mean().reset_index()

    df = avr_ges_nolink
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["ge"],df["tc"],label="tc",color="tab:blue")
    ax1.plot(df["ge"],df["tl"],label="tl",color="tab:orange")
    # ax1.plot(df["ge"],df["tf"],label="tf",color="tab:green")
    ax1.set_ylim(-0.09,1.19)
    ax1.set_xlabel("generation")
    # ax2 = ax1.twinx()
    # ax2.bar(df["ge"],df["ln"],color='lightblue',label="Link")
    # ax2.set_ylim(0,100)
    h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1, l1 ,loc="upper right")
    ax1.set_zorder(1)
    # ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    plt.savefig(name+"/"+name+"_nolink"+".png")
    plt.close()

for name in names:
    Graph_avr_tc_tl(csv=name+"/"+name+"_all.csv", name=name)