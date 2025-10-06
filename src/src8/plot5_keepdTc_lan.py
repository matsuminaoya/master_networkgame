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
from adjustText import adjust_text

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


# データを整理
data = {
    "10%": {1000: 1.0543,   3000: 1.0477, 5000: 1.0481},
    "30%": {1000: 1.0457,   3000: 1.0459, 5000: 1.0560},
    "50%": {1000: 1.0540,   3000: 1.0432, 5000: 1.0345},
    "70%": {1000: 1.0600,   3000: 1.0320, 5000: 1.0590},
    "90%": {1000: 1.0596,   3000: 1.0439, 5000: 1.0680},
}

x_values = [1000, 3000, 5000]

# グラフを作成
plt.figure(figsize=(6, 5))

for label, values in data.items():
    y_values = [values[x] for x in x_values]
    plt.plot(x_values, y_values, marker='o', label=label)

plt.title('leave an')
plt.xlabel('work')
plt.ylabel('tc')
plt.ylim(0.0, 1.1)
plt.legend(title='dencity')
plt.xticks(x_values)
plt.savefig("pic82/plot5_keepdTc_lan.png")
plt.show()