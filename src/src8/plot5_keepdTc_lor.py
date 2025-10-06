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

# データを整理
data = {
    "10%": {1000: 1.0051, 3000: 0.7171, 5000: 0.1360},
    "30%": {1000: 1.0616, 3000: 1.0200, 5000: 0.9390},
    "50%": {1000: 1.0572, 3000: 1.0275, 5000: 1.0539},
    "70%": {1000: 1.0456, 3000: 1.0378, 5000: 1.0355},
    "90%": {1000: 1.0562, 3000: 1.0760, 5000: 1.0414},
}

x_values = [1000, 3000, 5000]

# グラフを作成
plt.figure(figsize=(6, 5))

for label, values in data.items():
    y_values = [values[x] for x in x_values]
    plt.plot(x_values, y_values, marker='o', label=label)

plt.title('leave or')
plt.xlabel('work')
plt.ylabel('tc')
plt.ylim(0.0, 1.1)
plt.legend(title='dencity')
plt.xticks(x_values)
plt.savefig("pic82/plot5_keepdTc_lor.png")
plt.show()