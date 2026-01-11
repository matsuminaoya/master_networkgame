# tc*link network 有り無し プロット図
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
# name:6_both_full_noreset_t10_g10000_r100_w5000_b1
# tc: 0.5848
# ln: 94.916
# name:6_both_full_reset_t10_g10000_r100_w5000_b1
# tc: 0.8991999999999999
# ln: 13.954


# データ
names = [
    "Both / Null / NoReset", "Both / Null / Reset",
    "Form / Null / NoReset", "Form / Null / Reset",
    "Leave / Full / NoReset", "Leave / Full / Reset",
    "Both / Full / NoReset", "Both / Full / Reset"
]

tc_values = [0.5809, 0.0205, 1.0391, 0.2062, 0.5504, 0.7686, 0.5848, 0.8992]
ln_values = [95.914, 91.686, 99.0, 87.102, 0.0, 0.418, 94.916, 13.954]

# 色とマーカー
colors = ["red","red","green","green","orange","orange","b","b"]
markers = ['x', 'o', 'x', 'o','x', 'o', 'x', 'o',]

# プロット開始
plt.figure(figsize=(10, 6))

for name, x, y, c, m in zip(names, ln_values, tc_values, colors, markers):
    plt.scatter(x, y, color=c, marker=m, s=100, label=name)
    # plt.text(x + 3, y, name, fontsize=8, va='center')  # → 横にラベル、縦揃え

# 軸設定
plt.xlabel("link",fontsize=20)
plt.ylabel("tc",fontsize=20)
#plt.title("tc vs ln",)
plt.xlim(0, 100)
plt.ylim(0.0, 1.1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

handles, labels = plt.gca().get_legend_handles_labels()
order = [4, 5, 2, 3, 0, 1, 6, 7]  # 表示したい順
plt.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=18,
    frameon=True
)



plt.tight_layout()
plt.savefig("pic82/plot5_LinkTc_net_onlyboth.png")

#plt.show()