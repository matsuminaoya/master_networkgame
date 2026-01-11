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

names = ["7_both_null_noreset_t10_g10000_r100_w5000_b1",
         "7_both_null_reset_t10_g10000_r100_w5000_b1",
         "7_form_null_noreset_t10_g10000_r100_w5000_b1",
         "7_form_null_reset_t10_g10000_r100_w5000_b1",
         "7_leave_full_noreset_t10_g10000_r100_w5000_b1",
         "7_leave_full_reset_t10_g10000_r100_w5000_b1",
         "7_both_full_noreset_t10_g10000_r100_w5000_b1",
         "7_both_full_reset_t10_g10000_r100_w5000_b1",

         "7_oror_form_null_noreset_t10_g10000_r100_w5000_b1",
         "7_anan_leave_full_noreset_t10_g10000_r100_w5000_b1",
         "7_oror_both_null_noreset_t10_g10000_r100_w5000_b1",
         "7_anan_both_null_noreset_t10_g10000_r100_w5000_b1",
         "7_anor_both_null_noreset_t10_g10000_r100_w5000_b1",
         ] #TODO:

def main(csv, name, generation):
    data =pd.read_csv(csv)
    data_step = data[data["Generation"] == generation]
    
    tc_mean = data_step["tc"].mean()
    link_count_mean = data_step["link_count"].mean()

    print(f"name:{name}")
    print(f"tc: {tc_mean}")
    print(f"ln: {link_count_mean}")

for name in names:
    main("D:/master/src8/"+name+".csv", name=name, generation=10000) #TODO:

# name:7_both_null_noreset_t10_g10000_r100_w5000_b1
# tc: 0.5808999999999999
# ln: 95.914
# name:7_both_null_reset_t10_g10000_r100_w5000_b1
# tc: 0.0205
# ln: 91.686
# name:7_form_null_noreset_t10_g10000_r100_w5000_b1
# tc: 1.0391
# ln: 99.0
# name:7_form_null_reset_t10_g10000_r100_w5000_b1
# tc: 0.20619999999999997
# ln: 87.102
# name:7_leave_full_noreset_t10_g10000_r100_w5000_b1
# tc: 0.5504000000000001
# ln: 0.0
# name:7_leave_full_reset_t10_g10000_r100_w5000_b1
# tc: 0.7686
# ln: 0.418
# name:7_both_full_noreset_t10_g10000_r100_w5000_b1
# tc: 0.5848
# ln: 94.916
# name:7_both_full_reset_t10_g10000_r100_w5000_b1
# tc: 0.8991999999999999
# ln: 13.954

# name:7_oror_form_null_noreset_t10_g10000_r100_w5000_b1
# tc: 1.0632999999999997
# ln: 99.0
# name:7_anan_leave_full_noreset_t10_g10000_r100_w5000_b1
# tc: 0.5728
# ln: 0.0
# name:7_oror_both_null_noreset_t10_g10000_r100_w5000_b1
# tc: 1.037
# ln: 96.524
# name:7_anan_both_null_noreset_t10_g10000_r100_w5000_b1
# tc: 1.043
# ln: 88.89
# name:7_anor_both_null_noreset_t10_g10000_r100_w5000_b1
# tc: 1.0383999999999998
# ln: 62.352