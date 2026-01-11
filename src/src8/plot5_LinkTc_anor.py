# link*tc anor4タイプ プロット図
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

# データ
names = [
    "OR-AND",
    "OR-OR", "AND-AND", "AND-OR"
]

tc_values = [0.5809, 1.037, 1.043, 1.0383999999999998]
ln_values = [95.914, 96.524, 88.89, 62.352]

# 色とマーカー

colors = ["b","b","b","b"]
markers = ['o', 'x', 's', '^']

# プロット開始
plt.figure(figsize=(9, 6))

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
order = [0,1,2,3]  # 表示したい順
plt.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=18,
    frameon=True
)

plt.tight_layout()
plt.savefig("pic82/plot5_LinkTc_anor.png")

#plt.show()