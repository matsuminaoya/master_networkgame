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

names = [
         "3_keepd_09_form_or_t10_g10000_r100_w10000_b1",
         "3_keepd_07_form_or_t10_g10000_r100_w10000_b1",
         "3_keepd_05_form_or_t10_g10000_r100_w10000_b1",
         "3_keepd_03_form_or_t10_g10000_r100_w10000_b1",
         "3_keepd_01_form_or_t10_g10000_r100_w10000_b1",

         "3_keepd_09_form_or_t10_g10000_r100_w5000_b1",
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


         "3_keepd_09_leave_or_t10_g10000_r100_w10000_b1",
         "3_keepd_07_leave_or_t10_g10000_r100_w10000_b1",
         "3_keepd_05_leave_or_t10_g10000_r100_w10000_b1",
         "3_keepd_03_leave_or_t10_g10000_r100_w10000_b1",
         "3_keepd_01_leave_or_t10_g10000_r100_w10000_b1",

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


         "3_keepd_09_form_an_t10_g10000_r100_w10000_b1",
         "3_keepd_07_form_an_t10_g10000_r100_w10000_b1",
         "3_keepd_05_form_an_t10_g10000_r100_w10000_b1",
         "3_keepd_03_form_an_t10_g10000_r100_w10000_b1",
         "3_keepd_01_form_an_t10_g10000_r100_w10000_b1",

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


         "3_keepd_09_leave_an_t10_g10000_r100_w10000_b1",
         "3_keepd_07_leave_an_t10_g10000_r100_w10000_b1",
         "3_keepd_05_leave_an_t10_g10000_r100_w10000_b1",
         "3_keepd_03_leave_an_t10_g10000_r100_w10000_b1",
         "3_keepd_01_leave_an_t10_g10000_r100_w10000_b1",

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

# name:3_keepd_09_form_or_t10_g10000_r100_w10000_b1
# tc: 1.0445
# name:3_keepd_07_form_or_t10_g10000_r100_w10000_b1
# tc: 1.0565
# name:3_keepd_05_form_or_t10_g10000_r100_w10000_b1
# tc: 1.0700999999999998
# name:3_keepd_03_form_or_t10_g10000_r100_w10000_b1
# tc: 1.0598999999999998
# name:3_keepd_01_form_or_t10_g10000_r100_w10000_b1
# tc: 1.0343
# name:3_keepd_09_form_or_t10_g10000_r100_w5000_b1
# tc: 1.0298
# name:3_keepd_07_form_or_t10_g10000_r100_w5000_b1
# tc: 1.0720999999999998
# name:3_keepd_05_form_or_t10_g10000_r100_w5000_b1
# tc: 1.0520999999999998
# name:3_keepd_03_form_or_t10_g10000_r100_w5000_b1
# tc: 1.0566999999999998
# name:3_keepd_01_form_or_t10_g10000_r100_w5000_b1
# tc: 1.042
# name:3_keepd_09_form_or_t10_g10000_r100_w3000_b1
# tc: 1.058
# name:3_keepd_07_form_or_t10_g10000_r100_w3000_b1
# tc: 1.0403
# name:3_keepd_05_form_or_t10_g10000_r100_w3000_b1
# tc: 1.0628000000000002
# name:3_keepd_03_form_or_t10_g10000_r100_w3000_b1
# tc: 1.0555999999999999
# name:3_keepd_01_form_or_t10_g10000_r100_w3000_b1
# tc: 1.0612999999999997
# name:3_keepd_09_form_or_t10_g10000_r100_w1000_b1
# tc: 1.0492999999999997
# name:3_keepd_07_form_or_t10_g10000_r100_w1000_b1
# tc: 1.0553
# name:3_keepd_05_form_or_t10_g10000_r100_w1000_b1
# tc: 1.0418
# name:3_keepd_03_form_or_t10_g10000_r100_w1000_b1
# tc: 1.049
# name:3_keepd_01_form_or_t10_g10000_r100_w1000_b1
# tc: 1.0396999999999998


# name:3_keepd_09_leave_or_t10_g10000_r100_w10000_b1
# tc: 1.0399
# name:3_keepd_07_leave_or_t10_g10000_r100_w10000_b1
# tc: 1.0463999999999998
# name:3_keepd_05_leave_or_t10_g10000_r100_w10000_b1
# tc: 1.0531
# name:3_keepd_03_leave_or_t10_g10000_r100_w10000_b1
# tc: 0.8232999999999999
# name:3_keepd_01_leave_or_t10_g10000_r100_w10000_b1
# tc: 0.1426
# name:3_keepd_09_leave_or_t10_g10000_r100_w5000_b1
# tc: 1.0413999999999999
# name:3_keepd_07_leave_or_t10_g10000_r100_w5000_b1
# tc: 1.0355
# name:3_keepd_05_leave_or_t10_g10000_r100_w5000_b1
# tc: 1.0538999999999998
# name:3_keepd_03_leave_or_t10_g10000_r100_w5000_b1
# tc: 0.9389999999999997
# name:3_keepd_01_leave_or_t10_g10000_r100_w5000_b1
# tc: 0.13599999999999998
# name:3_keepd_09_leave_or_t10_g10000_r100_w3000_b1
# tc: 1.076
# name:3_keepd_07_leave_or_t10_g10000_r100_w3000_b1
# tc: 1.0378
# name:3_keepd_05_leave_or_t10_g10000_r100_w3000_b1
# tc: 1.0275
# name:3_keepd_03_leave_or_t10_g10000_r100_w3000_b1
# tc: 1.02
# name:3_keepd_01_leave_or_t10_g10000_r100_w3000_b1
# tc: 0.7171
# name:3_keepd_09_leave_or_t10_g10000_r100_w1000_b1
# tc: 1.0561999999999998
# name:3_keepd_07_leave_or_t10_g10000_r100_w1000_b1
# tc: 1.0455999999999999
# name:3_keepd_05_leave_or_t10_g10000_r100_w1000_b1
# tc: 1.0572
# name:3_keepd_03_leave_or_t10_g10000_r100_w1000_b1
# tc: 1.0615999999999999
# name:3_keepd_01_leave_or_t10_g10000_r100_w1000_b1
# tc: 1.0051


# name:3_keepd_09_form_an_t10_g10000_r100_w10000_b1
# tc: 1.0363999999999998
# name:3_keepd_07_form_an_t10_g10000_r100_w10000_b1
# tc: 1.067
# name:3_keepd_05_form_an_t10_g10000_r100_w10000_b1
# tc: 1.0163
# name:3_keepd_03_form_an_t10_g10000_r100_w10000_b1
# tc: 1.0001
# name:3_keepd_01_form_an_t10_g10000_r100_w10000_b1
# tc: 0.13679999999999998
# name:3_keepd_09_form_an_t10_g10000_r100_w5000_b1
# tc: 1.0478
# name:3_keepd_07_form_an_t10_g10000_r100_w5000_b1
# tc: 1.04
# name:3_keepd_05_form_an_t10_g10000_r100_w5000_b1
# tc: 1.0395999999999999
# name:3_keepd_03_form_an_t10_g10000_r100_w5000_b1
# tc: 0.9846999999999998
# name:3_keepd_01_form_an_t10_g10000_r100_w5000_b1
# tc: 0.038299999999999994
# name:3_keepd_09_form_an_t10_g10000_r100_w3000_b1
# tc: 1.0366999999999997
# name:3_keepd_07_form_an_t10_g10000_r100_w3000_b1
# tc: 1.0486
# name:3_keepd_05_form_an_t10_g10000_r100_w3000_b1
# tc: 1.0425
# name:3_keepd_03_form_an_t10_g10000_r100_w3000_b1
# tc: 1.0105
# name:3_keepd_01_form_an_t10_g10000_r100_w3000_b1
# tc: 0.1378
# name:3_keepd_09_form_an_t10_g10000_r100_w1000_b1
# tc: 1.0384
# name:3_keepd_07_form_an_t10_g10000_r100_w1000_b1
# tc: 1.0301
# name:3_keepd_05_form_an_t10_g10000_r100_w1000_b1
# tc: 1.0461999999999998
# name:3_keepd_03_form_an_t10_g10000_r100_w1000_b1
# tc: 1.0094
# name:3_keepd_01_form_an_t10_g10000_r100_w1000_b1
# tc: 0.9071999999999999


# name:3_keepd_09_leave_an_t10_g10000_r100_w10000_b1
# tc: 1.0371
# name:3_keepd_07_leave_an_t10_g10000_r100_w10000_b1
# tc: 1.0416999999999998
# name:3_keepd_05_leave_an_t10_g10000_r100_w10000_b1
# tc: 1.0592
# name:3_keepd_03_leave_an_t10_g10000_r100_w10000_b1
# tc: 1.0483999999999998
# name:3_keepd_01_leave_an_t10_g10000_r100_w10000_b1
# tc: 1.0639
# name:3_keepd_09_leave_an_t10_g10000_r100_w5000_b1
# tc: 1.068
# name:3_keepd_07_leave_an_t10_g10000_r100_w5000_b1
# tc: 1.059
# name:3_keepd_05_leave_an_t10_g10000_r100_w5000_b1
# tc: 1.0345
# name:3_keepd_03_leave_an_t10_g10000_r100_w5000_b1
# tc: 1.056
# name:3_keepd_01_leave_an_t10_g10000_r100_w5000_b1
# tc: 1.0480999999999998
# name:3_keepd_09_leave_an_t10_g10000_r100_w3000_b1
# tc: 1.0439
# name:3_keepd_07_leave_an_t10_g10000_r100_w3000_b1
# tc: 1.032
# name:3_keepd_05_leave_an_t10_g10000_r100_w3000_b1
# tc: 1.0432
# name:3_keepd_03_leave_an_t10_g10000_r100_w3000_b1
# tc: 1.0458999999999996
# name:3_keepd_01_leave_an_t10_g10000_r100_w3000_b1
# tc: 1.0476999999999999
# name:3_keepd_09_leave_an_t10_g10000_r100_w1000_b1
# tc: 1.0595999999999999
# name:3_keepd_07_leave_an_t10_g10000_r100_w1000_b1
# tc: 1.06
# name:3_keepd_05_leave_an_t10_g10000_r100_w1000_b1
# tc: 1.054
# name:3_keepd_03_leave_an_t10_g10000_r100_w1000_b1
# tc: 1.0456999999999999
# name:3_keepd_01_leave_an_t10_g10000_r100_w1000_b1
# tc: 1.0543