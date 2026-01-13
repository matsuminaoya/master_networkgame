# work*tc dencity別 折れ線グラフ
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

# データを整理
data = {
    "10%": {1000: 1.0397, 3000: 1.0613, 5000: 1.0420, 10000: 1.0343},
    "30%": {1000: 1.0490, 3000: 1.0556, 5000: 1.0567, 10000: 1.0599},
    "50%": {1000: 1.0418, 3000: 1.0628, 5000: 1.0521, 10000: 1.0701},
    "70%": {1000: 1.0553, 3000: 1.0403, 5000: 1.0721, 10000: 1.0565},
    "90%": {1000: 1.0493, 3000: 1.0580, 5000: 1.0298, 10000: 1.0445}
}

x_values = [1000, 3000, 5000, 10000]

# グラフを作成
plt.figure(figsize=(6, 5))

for label, values in data.items():
    y_values = [values[x] for x in x_values]
    plt.plot(x_values, y_values, marker='o', linewidth=3, markersize=8, label=label)

#plt.title('form an')
# 軸ラベル
plt.xlabel('w', fontsize=18)
plt.ylabel('tc', fontsize=18)

# 目盛り（メモリ）
plt.xticks(x_values, fontsize=18)
plt.yticks(fontsize=18)

# 範囲
plt.ylim(0.0, 1.1)

# 凡例
plt.legend(
    title='density',
    fontsize=18,
    title_fontsize=18,
    frameon=True
)

plt.tight_layout()
plt.savefig("pic82/plot5_keepdTc_for.png")
plt.show()