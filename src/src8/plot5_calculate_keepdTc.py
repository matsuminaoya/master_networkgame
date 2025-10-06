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

names = ["3_keepd_09_form_or_t10_g10000_r100_w5000_b1",
         "3_keepd_07_form_or_t10_g10000_r100_w5000_b1",
         "3_keepd_05_form_or_t10_g10000_r100_w5000_b1",
         "3_keepd_03_form_or_t10_g10000_r100_w5000_b1",
         "3_keepd_01_form_or_t10_g10000_r100_w5000_b1",
         
         "3_keepd_09_form_or_t10_g10000_r100_w3000_b1",
         "3_keepd_07_form_or_t10_g10000_r100_w3000_b1",
         "3_keepd_05_form_or_t10_g10000_r100_w3000_b1",
         "3_keepd_03_form_or_t10_g10000_r100_w3000_b1",
         "3_keepd_01_form_or_t10_g10000_r100_w3000_b1",

         "3_keepd_09_form_or_t10_g10000_r100_w1000_b1",
         "3_keepd_07_form_or_t10_g10000_r100_w1000_b1",
         "3_keepd_05_form_or_t10_g10000_r100_w1000_b1",
         "3_keepd_03_form_or_t10_g10000_r100_w1000_b1",
         "3_keepd_01_form_or_t10_g10000_r100_w1000_b1",


         "3_keepd_09_leave_or_t10_g10000_r100_w5000_b1",
         "3_keepd_07_leave_or_t10_g10000_r100_w5000_b1",
         "3_keepd_05_leave_or_t10_g10000_r100_w5000_b1",
         "3_keepd_03_leave_or_t10_g10000_r100_w5000_b1",
         "3_keepd_01_leave_or_t10_g10000_r100_w5000_b1",

         "3_keepd_09_leave_or_t10_g10000_r100_w3000_b1",
         "3_keepd_07_leave_or_t10_g10000_r100_w3000_b1",
         "3_keepd_05_leave_or_t10_g10000_r100_w3000_b1",
         "3_keepd_03_leave_or_t10_g10000_r100_w3000_b1",
         "3_keepd_01_leave_or_t10_g10000_r100_w3000_b1",

         "3_keepd_09_leave_or_t10_g10000_r100_w1000_b1",
         "3_keepd_07_leave_or_t10_g10000_r100_w1000_b1",
         "3_keepd_05_leave_or_t10_g10000_r100_w1000_b1",
         "3_keepd_03_leave_or_t10_g10000_r100_w1000_b1",
         "3_keepd_01_leave_or_t10_g10000_r100_w1000_b1",


         "3_keepd_09_form_an_t10_g10000_r100_w5000_b1",
         "3_keepd_07_form_an_t10_g10000_r100_w5000_b1",
         "3_keepd_05_form_an_t10_g10000_r100_w5000_b1",
         "3_keepd_03_form_an_t10_g10000_r100_w5000_b1",
         "3_keepd_01_form_an_t10_g10000_r100_w5000_b1",
         
         "3_keepd_09_form_an_t10_g10000_r100_w3000_b1",
         "3_keepd_07_form_an_t10_g10000_r100_w3000_b1",
         "3_keepd_05_form_an_t10_g10000_r100_w3000_b1",
         "3_keepd_03_form_an_t10_g10000_r100_w3000_b1",
         "3_keepd_01_form_an_t10_g10000_r100_w3000_b1",

         "3_keepd_09_form_an_t10_g10000_r100_w1000_b1",
         "3_keepd_07_form_an_t10_g10000_r100_w1000_b1",
         "3_keepd_05_form_an_t10_g10000_r100_w1000_b1",
         "3_keepd_03_form_an_t10_g10000_r100_w1000_b1",
         "3_keepd_01_form_an_t10_g10000_r100_w1000_b1",


         "3_keepd_09_leave_an_t10_g10000_r100_w5000_b1",
         "3_keepd_07_leave_an_t10_g10000_r100_w5000_b1",
         "3_keepd_05_leave_an_t10_g10000_r100_w5000_b1",
         "3_keepd_03_leave_an_t10_g10000_r100_w5000_b1",
         "3_keepd_01_leave_an_t10_g10000_r100_w5000_b1",
         
         "3_keepd_09_leave_an_t10_g10000_r100_w3000_b1",
         "3_keepd_07_leave_an_t10_g10000_r100_w3000_b1",
         "3_keepd_05_leave_an_t10_g10000_r100_w3000_b1",
         "3_keepd_03_leave_an_t10_g10000_r100_w3000_b1",
         "3_keepd_01_leave_an_t10_g10000_r100_w3000_b1",

         "3_keepd_09_leave_an_t10_g10000_r100_w1000_b1",
         "3_keepd_07_leave_an_t10_g10000_r100_w1000_b1",
         "3_keepd_05_leave_an_t10_g10000_r100_w1000_b1",
         "3_keepd_03_leave_an_t10_g10000_r100_w1000_b1",
         "3_keepd_01_leave_an_t10_g10000_r100_w1000_b1",
         ] #TODO:

def main(csv, name, generation):
    data =pd.read_csv(csv)
    data_step = data[data["Generation"] == generation]
    
    tc_mean = data_step["tc"].mean()

    print(f"name:{name}")
    print(f"tc: {tc_mean}")

for name in names:
    main("D:/master/src8/"+name+".csv", name=name, generation=10000) #TODO:

