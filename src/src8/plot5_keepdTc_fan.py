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

# データを整理
data = {
    "10%": {1000: 0.9072, 3000: 0.1378, 5000: 0.0383, 10000: 0.1368},
    "30%": {1000: 1.0094, 3000: 1.0105, 5000: 0.9847, 10000: 1.0001},
    "50%": {1000: 1.0462, 3000: 1.0425, 5000: 1.0396, 10000: 1.0163},
    "70%": {1000: 1.0301, 3000: 1.0486, 5000: 1.0400, 10000: 1.0670},
    "90%": {1000: 1.0384, 3000: 1.0367, 5000: 1.0478, 10000: 1.0364}
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
plt.savefig("pic82/plot5_keepdTc_fan.png")
plt.show()