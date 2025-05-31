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

names = ["6_anan_leave_full_noreset_t10_g10000_r100_w5000_b1",] #TODO:

def Graph_avr_tc_tf(csv, name):
    #allavr
    data =pd.read_csv(csv)
    #data_step = data[data["Generation"] % 100 ==0]
    avr_ges_all = data.groupby("Generation")[["tc","tl","link_count"]].mean().reset_index()
    
    df = avr_ges_all
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["Generation"],df["tc"],label="tc",color="tab:blue")
    ax1.plot(df["Generation"],df["tl"],label="tl",color="tab:orange")
    ax1.set_ylim(0.0,1.1)
    ax1.set_xlabel("generation")
    ax1.set_ylabel("tc, tl")
    ax2 = ax1.twinx()
    ax2.bar(df["Generation"],df["link_count"],color='lightblue',label="link")
    ax2.set_ylim(0.0,100.0)
    ax2.set_ylabel("link")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc="upper right")
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    plt.savefig("pic8/"+name+"_allavr5.png") #TODO:
    plt.close()

    # #linked
    # data_linked = data_step[data_step["ln"] > 0]
    # avr_ges_linked = data_linked.groupby("ge")[["tc","tf"]].mean().reset_index()

    # df = avr_ges_linked
    # fig = plt.figure(figsize=(20,10))
    # ax1 = fig.add_subplot()
    # ax1.plot(df["ge"],df["tc"],label="tc",color="tab:blue")
    # # ax1.plot(df["ge"],df["tl"],label="tl",color="tab:orange")
    # ax1.plot(df["ge"],df["tf"],label="tf",color="tab:green")
    # ax1.set_ylim(-0.09,1.19)
    # ax1.set_xlabel("generation")
    # # ax2 = ax1.twinx()
    # # ax2.bar(df["ge"],df["ln"],color='lightblue',label="Link")
    # # ax2.set_ylim(0,100)
    # h1, l1 = ax1.get_legend_handles_labels()
    # # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1, l1 ,loc="upper right")
    # ax1.set_zorder(1)
    # # ax2.set_zorder(0)
    # ax1.patch.set_alpha(0)
    # plt.savefig(name+"/"+name+"_linked_100"+".png")
    # plt.close()

    # #nolink
    # data_nolink = data_step[data_step["ln"] == 0]
    # avr_ges_nolink = data_nolink.groupby("ge")[["tc","tf"]].mean().reset_index()

    # df = avr_ges_nolink
    # fig = plt.figure(figsize=(20,10))
    # ax1 = fig.add_subplot()
    # ax1.plot(df["ge"],df["tc"],label="tc",color="tab:blue")
    # # ax1.plot(df["ge"],df["tl"],label="tl",color="tab:orange")
    # ax1.plot(df["ge"],df["tf"],label="tf",color="tab:green")
    # ax1.set_ylim(-0.09,1.19)
    # ax1.set_xlabel("generation")
    # # ax2 = ax1.twinx()
    # # ax2.bar(df["ge"],df["ln"],color='lightblue',label="Link")
    # # ax2.set_ylim(0,100)
    # h1, l1 = ax1.get_legend_handles_labels()
    # # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1, l1 ,loc="upper right")
    # ax1.set_zorder(1)
    # # ax2.set_zorder(0)
    # ax1.patch.set_alpha(0)
    # plt.savefig(name+"/"+name+"_nolink_100"+".png")
    # plt.close()

for name in names:
    Graph_avr_tc_tf(csv="D:/master/src8/ouchi/"+name+".csv", name=name) #TODO: