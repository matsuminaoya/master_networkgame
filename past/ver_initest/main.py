#python.exe -m pip install --upgrade pip
#import gc
import time
import random
import inspect
import numpy as np #pip install numpy
import pandas as pd #pip install pandas
import seaborn as sns #pip install seaborn
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx #pip install networkx
time0 = time.time()

#grobal value
n = 100 #100 #4
bam = 2 #banetwork'S new node's link number
a = 1.0
bene = 2.0
cost = 1.0
mutation = 0.01
#tr>ge>ro>work
trial = 1 #1,10 #2
generation = 5001 #5001 #3 #mod(ani_step)=1ないとエラーが発生するので注意
roound = 100 #100
work = 5000 #5000
#graph_step
g_step = 100 #100
ani_step = 100 #100

#for testing
# n = 3
# trial = 2
# generation = 51
# roound = 2
# work = 2
# g_step = 10
# ani_step = 10

#define
def Randomn(): #ok
    rng = np.random.default_rng()
    return rng.random(n)

def Sigmoid(x): #ok
    return np.exp(np.minimum(x,0)) / (1 + np.exp(-np.abs(x)))

def Initialize_value(): #ok
    rng = np.random.default_rng()
    return 0.1*rng.integers(12,size=(1,n))[0]

def Initialize_linkmatrix_full(): #ok
    linkmatrix_rvs = np.identity(n)
    return np.where(linkmatrix_rvs==0,1,0)
def Initialize_linkmatrix_null(): #ok
    return np.zeros((n,n),dtype=np.int16)
#def ba done

def Calculate_cnum(coop_ro, linkmatrix): #ok
    coop_ro_rvs = np.where(coop_ro==0,1,0) #ok
    noncoop_index_ro = np.nonzero(coop_ro_rvs) #ok
    linkmatrix_del = np.delete(linkmatrix, noncoop_index_ro, 1) #非協力者の列削除
    cnum_ro = np.sum(linkmatrix_del,axis=1)
    return cnum_ro
def Calculate_poff_ro(coop_ro, lnum_ro, cnum_ro): #ok #利得は人数で割ってる
    poff_ro_accu = np.where(coop_ro==1, (cnum_ro*bene)-(lnum_ro*cost), cnum_ro*bene)
    poff_ro = np.divide(poff_ro_accu, lnum_ro, out=poff_ro_accu, where=(lnum_ro!=0))
    return poff_ro

# def Initialize_poff_full(coop_ro, cnum_ro): #ok
#     return np.where(coop_ro==1, cnum_ro*bene-(n-1)*cost, cnum_ro*bene)
# def Initialize_poff_null(): #ok
#     return np.zeros((1,n))[0]

# def Initialize_count_game_ge_full(): #ok
#     return np.ones((1,n))[0]
# def Initialize_count_game_ge_null(): #ok
#     return np.zeros((1,n))[0]

def Linked_choice(linkmatrix, count_poff_ge, cho): #ok #cho=[]
    linked_poff = linkmatrix*count_poff_ge #ok # TODO: ここあってるん？
    for k in range(n):
        linked_cho = linked_poff[k].nonzero()
        if len(linked_cho[0]) == 0: #自分をついか
            cho.append(k)
        else:
            cho_index = random.choice(linked_cho[0])
            cho.append(cho_index.item())
    return cho

def Selection_tc_tl_tf(m_random, count_poff_ge, cho, tc_pre, tl_pre, tf_pre): #okok
    fermi = Sigmoid(1.0*a*(count_poff_ge[cho]-count_poff_ge)) #引き算の大きさが影響してる？フェルミの傾き
    f_random = Randomn()
    tc = np.where(((mutation<=m_random)&(f_random<fermi)), tc_pre[cho], tc_pre)
    f_random = Randomn()
    tl = np.where(((mutation<=m_random)&(f_random<fermi)), tl_pre[cho], tl_pre)
    f_random = Randomn()
    tf = np.where(((mutation<=m_random)&(f_random<fermi)), tf_pre[cho], tf_pre)
    return tc,tl,tf
def Selection_tc_tl(m_random, count_poff_ge, cho, tc_pre, tl_pre): #okok
    fermi = Sigmoid(1.0*a*(count_poff_ge[cho]-count_poff_ge))
    f_random = Randomn()
    tc = np.where(((mutation<=m_random)&(f_random<fermi)), tc_pre[cho], tc_pre)
    f_random = Randomn()
    tl = np.where(((mutation<=m_random)&(f_random<fermi)), tl_pre[cho], tl_pre)
    return tc,tl
def Selection_tc_tf(m_random, count_poff_ge, cho, tc_pre, tf_pre): #okok based #imitation
    fermi = Sigmoid(1.0*a*(count_poff_ge[cho]-count_poff_ge))
    f_random = Randomn()
    tc = np.where(((mutation<=m_random)&(f_random<fermi)), tc_pre[cho], tc_pre)
    f_random = Randomn()
    tf = np.where(((mutation<=m_random)&(f_random<fermi)), tf_pre[cho], tf_pre)
    return tc,tf

def Mutation_tc_tl_tf(m_random, tc_pre, tl_pre, tf_pre): #okok
    plus_minus = Randomn()
    tc_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tc_pre<1.1)), tc_pre+0.1, tc_pre)
    tc = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tc_pre>0)), tc_pre-0.1, tc_1)
    plus_minus = Randomn()
    tl_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tl_pre<1.1)), tl_pre+0.1, tl_pre)
    tl = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tl_pre>0)), tl_pre-0.1, tl_1)
    plus_minus = Randomn()
    tf_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tf_pre<1.1)), tf_pre+0.1, tf_pre)
    tf = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tf_pre>0)), tf_pre-0.1, tf_1)
    return tc,tl,tf
def Mutation_tc_tl(m_random, tc_pre, tl_pre): #okok
    plus_minus = Randomn()
    tc_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tc_pre<1.1)), tc_pre+0.1, tc_pre)
    tc = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tc_pre>0)), tc_pre-0.1, tc_1)
    plus_minus = Randomn()
    tl_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tl_pre<1.1)), tl_pre+0.1, tl_pre)
    tl = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tl_pre>0)), tl_pre-0.1, tl_1)
    return tc,tl
def Mutation_tc_tf(m_random, tc_pre, tf_pre): #okok based
    plus_minus = Randomn()
    tc_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tc_pre<1.1)), tc_pre+0.1, tc_pre)
    tc = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tc_pre>0)), tc_pre-0.1, tc_1)
    plus_minus = Randomn()
    tf_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tf_pre<1.1)), tf_pre+0.1, tf_pre)
    tf = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tf_pre>0)), tf_pre-0.1, tf_1)
    return tc,tf

def Coop_ro_zero(tc): #ok
    tc_random = Randomn()
    return np.where((tc<=tc_random),1,0)
def Coop_ro_nonzero(cnum_ro, lnum_ro, tc): #okok
    tc_random = Randomn()
    ratio_clink = np.divide(cnum_ro, lnum_ro, where=lnum_ro>0)
    coop_ro_1 = np.where(((lnum_ro>0)&(ratio_clink>=tc)), 1, 0)
    coop_ro_2 = np.where(((lnum_ro==0)&(tc<=tc_random)), 1, coop_ro_1)
    coop_ro = np.where(((lnum_ro==0)&(tc>tc_random)), 0, coop_ro_2)
    return coop_ro

def Leave_Form_tl_tf(linkmatrix, coop_ratio, tl, tf): #TODO: きりはりだとｔｃがさがる、ｇが少ないと協力的、ｇが100000、500、100・・・900,gが少ないとfullだと非協力的に、平均と累積べつべつ、
    rng = np.random.default_rng()
    pair_index = np.triu_indices(n, k=1)
    x = rng.integers(((n-1)*n/2),size=(1,work))[0]
    i = (pair_index[0][x])
    j = (pair_index[1][x])
    for k in range(work):
        pair = (i[k],j[k])
        mask_l = ((linkmatrix[pair]==1) & ((coop_ratio[pair[0]]<tl[pair[1]]) | (coop_ratio[pair[1]]<tl[pair[0]])))
        mask_f = ((linkmatrix[pair]==0) & ((coop_ratio[pair[0]]>=tf[pair[1]]) & (coop_ratio[pair[1]]>=tf[pair[0]])))
        linkmatrix[pair[0][mask_l],pair[1][mask_l]] = 0
        linkmatrix[pair[1][mask_l],pair[0][mask_l]] = 0
        linkmatrix[pair[0][mask_f],pair[1][mask_f]] = 1
        linkmatrix[pair[1][mask_f],pair[0][mask_f]] = 1
    return linkmatrix
def Leave_Form_tl(linkmatrix, coop_ratio, tl): #ok
    rng = np.random.default_rng()
    pair_index = np.triu_indices(n, k=1)
    x = rng.integers(((n-1)*n/2),size=(1,work))[0]
    i = (pair_index[0][x])
    j = (pair_index[1][x])
    pair = (i,j)
    mask_l = ((linkmatrix[pair]==1) & ((coop_ratio[pair[0]]<tl[pair[1]]) | (coop_ratio[pair[1]]<tl[pair[0]])))
    linkmatrix[pair[0][mask_l],pair[1][mask_l]] = 0
    linkmatrix[pair[1][mask_l],pair[0][mask_l]] = 0
    return linkmatrix
def Leave_Form_tf(linkmatrix, coop_ratio, tf): #ok
    rng = np.random.default_rng()
    pair_index = np.triu_indices(n, k=1)
    x = rng.integers(((n-1)*n/2),size=(1,work))[0]
    i = (pair_index[0][x])
    j = (pair_index[1][x])
    pair = (i,j)
    mask_f = ((linkmatrix[pair]==0) & ((coop_ratio[pair[0]]>=tf[pair[1]]) & (coop_ratio[pair[1]]>=tf[pair[0]])))
    linkmatrix[pair[0][mask_f],pair[1][mask_f]] = 1
    linkmatrix[pair[1][mask_f],pair[0][mask_f]] = 1
    return linkmatrix

def Graph_avr_tc_tl_tf(csv): #
    df =pd.read_csv(csv)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["ge"],df["tc"],label="tc")
    ax1.plot(df["ge"],df["tl"],label="tl")
    ax1.plot(df["ge"],df["tf"],label="tf")
    ax1.set_ylim(0,1.1)
    ax1.set_xlabel("generation")
    ax2 = ax1.twinx()
    ax2.bar(df["ge"],df["ln"],color="lightblue",label="lnum")
    ax2.set_ylim(0,100)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc="upper right")
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    print(inspect.currentframe().f_code.co_name)
    return plt
def Graph_avr_tc_tl(csv): #
    df =pd.read_csv(csv)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["ge"],df["tc"],label="tc")
    ax1.plot(df["ge"],df["tl"],label="tl")
    ax1.set_ylim(0,1.1)
    ax1.set_xlabel("generation")
    ax2 = ax1.twinx()
    ax2.bar(df["ge"],df["ln"],color="lightblue",label="lnum")
    ax2.set_ylim(0,100)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc="upper right")
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    print(inspect.currentframe().f_code.co_name)
    return plt
def Graph_avr_tc_tf(csv): #
    df = pd.read_csv(csv)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["ge"],df["tc"],label="tc")
    ax1.plot(df["ge"],df["tf"],label="tf")
    ax1.set_ylim(-0.09,1.19)
    ax1.set_xlabel("generation")
    ax2 = ax1.twinx()
    ax2.bar(df["ge"],df["ln"],color="lightblue",label="lnum")
    ax2.set_ylim(-0.09,100.09)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc="upper right")
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    print(inspect.currentframe().f_code.co_name)
    return plt

def Graph_all_tc_tl_tf_dfexplode(csv): #
    df = pd.read_csv(csv)
    df = df[df["ge"]%g_step==0]
    df["tc"] = df["tc"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["tl"] = df["tl"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["tf"] = df["tf"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["ln"] = df["ln"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df = df.explode(["tc", "tl", "tf", "ln"], ignore_index=True)
    df["tc"] = df["tc"].astype(float)
    df["tl"] = df["tl"].astype(float)
    df["tf"] = df["tf"].astype(float)
    df["ln"] = df["ln"].astype(float)
    print(inspect.currentframe().f_code.co_name)
    return df
def Graph_all_tc_tl_dfexplode(csv): #
    df = pd.read_csv(csv)
    df = df[df["ge"]%g_step==0]
    df["tc"] = df["tc"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["tl"] = df["tl"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["ln"] = df["ln"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df = df.explode(["tc", "tl", "ln"], ignore_index=True)
    df["tc"] = df["tc"].astype(float)
    df["tl"] = df["tl"].astype(float)
    print(inspect.currentframe().f_code.co_name)
    return df
def Graph_all_tc_tf_dfexplode(csv): #
    df = pd.read_csv(csv)
    df = df[df["ge"]%g_step==0]
    df["tc"] = df["tc"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["tf"] = df["tf"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df["ln"] = df["ln"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    df = df.explode(["tc", "tf", "ln"], ignore_index=True)
    df["tc"] = df["tc"].astype(float)
    df["tf"] = df["tf"].astype(float)
    df["ln"] = df["ln"].astype(float)
    print(inspect.currentframe().f_code.co_name)
    return df

def Graph_all_vio(df, ylabel): #
    plt.figure(figsize=(20,10))
    plt.ylabel(ylabel)
    plt.xlabel("ge")
    sns.violinplot(x="ge",y=ylabel,data=df)
    print(inspect.currentframe().f_code.co_name)
    return plt
def Graph_all_box(df, ylabel): #
    plt.figure(figsize=(20,10))
    if ylabel != "ln":
        plt.ylim(-0.009,1.109)
    else:
        plt.ylim(-0.09,100.09)
    plt.ylabel(ylabel)
    plt.xlabel("ge")
    sns.boxplot(x="ge", y=ylabel, data=df)
    print(inspect.currentframe().f_code.co_name)
    return plt
def Graph_network_ani(linkmatrix_ges): #
    # グラフの描画設定
    fig, ax = plt.subplots(figsize=(15,15))
    G = nx.from_numpy_array(linkmatrix_ges[0])  # 初期グラフを生成
    #pos = nx.circular_layout(G)  # ノードの位置を円状に固定
    pos = nx.spring_layout(G)
    def update(frame):
        ax.clear()  # 前のフレームをクリア
        G = nx.from_numpy_array(linkmatrix_ges[frame*ani_step])  # 隣接行列からグラフを生成
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, width=0.5, alpha=0.5)
        ax.set_title(f"ge {frame*ani_step}")
    # アニメーションの設定
    ani = FuncAnimation(fig, update, frames=int(generation/ani_step)+1, repeat=True, interval=1000)
    print(inspect.currentframe().f_code.co_name)
    return ani

def Elapsed_time_hms(elapsed_time): #ok
    time_h = int(elapsed_time / 3600)
    time_m = int(elapsed_time % 3600 / 60)
    time_s = int(elapsed_time % 60)
    return ":time: {0:02}:{1:02}:{2:02}".format(time_h, time_m, time_s)

def Plotly_network_ani(linkmatrix_ges): #future work
    korekarayaru = 1
    return

# print(
# Leave_Form(
# #coop_ro=np.array([1,0,1]),
# #cnum_ro=np.array([2,2,2])
# linkmatrix=np.array([[0,1,1],[1,0,1],[1,1,0]]),
# #count_poff_ge=np.array([4,9,1]),
# #cho=[1,2,0],
# #tc=np.array([0.0,0.1,0.2]),
# tl=np.array([0.8,0.8,0.8]),
# tf=np.array([0.1,0.1,0.1]),
# coop_ratio = np.array([0.5,0.5,0.5])
# )
# )

def start_bo_full():#ok
    name = inspect.currentframe().f_code.co_name
    tc_avr_ges_trs,tl_avr_ges_trs,tf_avr_ges_trs,ln_avr_ges_trs = [],[],[],[]
    tc_all_ges_trs,tl_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs = [],[],[],[]
    linkmatrix_ges_tr0 = []
    for tr in range(trial):
        tc = Initialize_value()
        tl = Initialize_value()
        tf = Initialize_value()
        linkmatrix = Initialize_linkmatrix_full()
        tc_avr_ges,tl_avr_ges,tf_avr_ges,ln_avr_ges = [],[],[],[]
        tc_all_ges,tl_all_ges,tf_all_ges,ln_all_ges = [],[],[],[]
        for ge in range(generation):
            for ro in range(roound):
                if ro == 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #追加,今回のリンク数を調べる
                    coop_ro = Coop_ro_zero(tc) #これだけで自分の協力非協力決める
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #変更
                    count_game_ge = np.where(lnum_ro>0, 1, 0) #変更
                    count_coop_game_ge = np.where((lnum_ro>0)&(coop_ro==1), 1, 0) #変更
                    count_poff_ge = poff_ro
                if ro > 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #今回のリンク数調べる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更前の協力数調べる
                    coop_ro = Coop_ro_nonzero(cnum_ro=cnum_ro,lnum_ro=lnum_ro, tc=tc) #それで自分の協力非協力きめる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #利得求める
                    count_game_ge += np.where(lnum_ro>0, 1, 0)
                    count_coop_game_ge += np.where((lnum_ro>0)&(coop_ro==1), 1, 0)
                    count_poff_ge += poff_ro
                if ro < roound-1:
                    coop_ratio = np.divide(count_coop_game_ge, count_game_ge, where=count_game_ge>0)
                    linkmatrix = Leave_Form_tl_tf(linkmatrix=linkmatrix,coop_ratio=coop_ratio, tl=tl, tf=tf)
            print(str(tr)+"tr-"+str(ge)+"ge")
            ln = np.sum(linkmatrix,axis=1)
            #sellection
            m_random = Randomn()
            cho = Linked_choice(linkmatrix, count_poff_ge, cho=[])
            tc,tl,tf = Selection_tc_tl_tf(m_random=m_random,count_poff_ge=count_poff_ge,cho=cho,tc_pre=tc,tl_pre=tl,tf_pre=tf)
            #mutation
            tc,tl,tf = Mutation_tc_tl_tf(m_random=m_random,tc_pre=tc,tl_pre=tl,tf_pre=tf)
            #graph
            tc_avr_ges.append(mean(tc)) #ok
            tl_avr_ges.append(mean(tl))
            tf_avr_ges.append(mean(tf))
            ln_avr_ges.append(mean(ln))
            tc_all_ges.append(tc)
            tl_all_ges.append(tl)
            tf_all_ges.append(tf)
            ln_all_ges.append(ln)
            if tr == 0:
                linkmatrix_ges_tr0.append(linkmatrix)
        tc_avr_ges_trs.append(tc_avr_ges) #ok
        tl_avr_ges_trs.append(tl_avr_ges)
        tf_avr_ges_trs.append(tf_avr_ges)
        ln_avr_ges_trs.append(ln_avr_ges)
        tc_all_ges_trs.extend(tc_all_ges)
        tl_all_ges_trs.extend(tl_all_ges)
        tf_all_ges_trs.extend(tf_all_ges)
        ln_all_ges_trs.extend(ln_all_ges)
    time1 = time.time()#new
    print("fin" + Elapsed_time_hms(elapsed_time=(time1-time0)))#new
    #oresen
    ge_ges = np.arange(generation)
    ln_avr_ges_trs_avr = np.mean(ln_avr_ges_trs, axis=0)
    tc_avr_ges_trs_avr = np.mean(tc_avr_ges_trs, axis=0)
    tl_avr_ges_trs_avr = np.mean(tl_avr_ges_trs, axis=0)
    tf_avr_ges_trs_avr = np.mean(tf_avr_ges_trs, axis=0)
    df = pd.DataFrame({"ge":ge_ges,"tc":tc_avr_ges_trs_avr,"tl":tl_avr_ges_trs_avr,"tf":tf_avr_ges_trs_avr,"ln":ln_avr_ges_trs_avr})
    df.to_csv(name + "_avr.csv")
    Graph_avr_tc_tl_tf(name + "_avr.csv").savefig(name + "_avr.png")
    #vio box
    tr_trs_repeat = np.repeat(np.arange(trial),generation)
    ge_ges_repeat = np.tile(ge_ges, trial)
    df = pd.DataFrame({"tr":tr_trs_repeat,"ge":ge_ges_repeat,"tc":tc_all_ges_trs,"tl":tl_all_ges_trs,"tf":tf_all_ges_trs,"ln":ln_all_ges_trs})
    df.to_csv(name + "_all.csv")
    df = Graph_all_tc_tl_tf_dfexplode(name + "_all.csv")
    # time2 = time.time()
    # Graph_all_vio(df, ylabel="tc").savefig(name + "_all_vio_tc.png")
    # time3 = time.time()
    # print("vio"+Elapsed_time_hms(time3-time2))
    # Graph_all_vio(df, ylabel="tl").savefig(name + "_all_vio_tl.png")
    # Graph_all_vio(df, ylabel="tf").savefig(name + "_all_vio_tf.png")
    # Graph_all_vio(df, ylabel="ln").savefig(name + "_all_vio_ln.png")
    time4 = time.time()
    Graph_all_box(df, ylabel="tc").savefig(name + "_all_box_tc.png")
    time5 = time.time()
    print("box"+Elapsed_time_hms(time5-time4))
    Graph_all_box(df, ylabel="tl").savefig(name + "_all_box_tl.png")
    Graph_all_box(df, ylabel="tf").savefig(name + "_all_box_tf.png")
    Graph_all_box(df, ylabel="ln").savefig(name + "_all_box_ln.png")
    df = pd.DataFrame({"ge":ge_ges, "linkmatrix":linkmatrix_ges_tr0})
    df.to_csv(name + "_tr0_network.csv")
    time6 = time.time()
    Graph_network_ani(linkmatrix_ges=linkmatrix_ges_tr0).save(name + "_tr0_network.gif", writer='pillow', fps=60)
    time7 = time.time()
    print("ani"+Elapsed_time_hms(time7-time6))
    print("all"+Elapsed_time_hms(time7-time0))
#start_bo_full()

def start_le_full():#ok
    name = inspect.currentframe().f_code.co_name
    tc_avr_ges_trs,tl_avr_ges_trs,ln_avr_ges_trs = [],[],[]
    tc_all_ges_trs,tl_all_ges_trs,ln_all_ges_trs = [],[],[]
    linkmatrix_ges_tr0 = []
    for tr in range(trial):
        tc = Initialize_value()
        tl = Initialize_value()
        linkmatrix = Initialize_linkmatrix_full()
        tc_avr_ges,tl_avr_ges,ln_avr_ges = [],[],[]
        tc_all_ges,tl_all_ges,ln_all_ges = [],[],[]
        for ge in range(generation):
            for ro in range(roound):
                if ro == 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #追加,今回のリンク数を調べる
                    coop_ro = Coop_ro_zero(tc) #これだけで自分の協力非協力決める
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #変更
                    count_game_ge = np.where(lnum_ro>0, 1, 0) #変更
                    count_coop_game_ge = np.where((lnum_ro>0)&(coop_ro==1), 1, 0) #変更
                    count_poff_ge = poff_ro
                if ro > 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #今回のリンク数調べる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更前の協力数調べる
                    coop_ro = Coop_ro_nonzero(cnum_ro=cnum_ro,lnum_ro=lnum_ro, tc=tc) #それで自分の協力非協力きめる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #利得求める
                    count_game_ge += np.where(lnum_ro>0, 1, 0)
                    count_coop_game_ge += np.where((lnum_ro>0)&(coop_ro==1), 1, 0)
                    count_poff_ge += poff_ro
                if ro < roound-1:
                    coop_ratio = np.divide(count_coop_game_ge, count_game_ge, where=count_game_ge>0)
                    linkmatrix = Leave_Form_tl(linkmatrix=linkmatrix,coop_ratio=coop_ratio, tl=tl)
            print(str(tr)+"tr-"+str(ge)+"ge")
            ln = np.sum(linkmatrix,axis=1)
            #sellection
            m_random = Randomn()
            cho = Linked_choice(linkmatrix, count_poff_ge, cho=[])
            tc,tl = Selection_tc_tl(m_random=m_random, count_poff_ge=count_poff_ge, cho=cho, tc_pre=tc, tl_pre=tl)
            #mutation
            tc,tl = Mutation_tc_tl(m_random=m_random,tc_pre=tc,tl_pre=tl)
            #graph
            tc_avr_ges.append(mean(tc)) #ok
            tl_avr_ges.append(mean(tl))
            ln_avr_ges.append(mean(ln))
            tc_all_ges.append(tc)
            tl_all_ges.append(tl)
            ln_all_ges.append(ln)
            if tr == 0:
                linkmatrix_ges_tr0.append(linkmatrix)
        tc_avr_ges_trs.append(tc_avr_ges) #ok
        tl_avr_ges_trs.append(tl_avr_ges)
        ln_avr_ges_trs.append(ln_avr_ges)
        tc_all_ges_trs.extend(tc_all_ges)
        tl_all_ges_trs.extend(tl_all_ges)
        ln_all_ges_trs.extend(ln_all_ges)
    time1 = time.time()#new
    print("sim"+Elapsed_time_hms(elapsed_time=(time1-time0)))#new
    #oresen
    ge_ges = np.arange(generation)
    ln_avr_ges_trs_avr = np.mean(ln_avr_ges_trs, axis=0)
    tc_avr_ges_trs_avr = np.mean(tc_avr_ges_trs, axis=0)
    tl_avr_ges_trs_avr = np.mean(tl_avr_ges_trs, axis=0)
    df = pd.DataFrame({"ge":ge_ges,"tc":tc_avr_ges_trs_avr,"tl":tl_avr_ges_trs_avr, "ln":ln_avr_ges_trs_avr})
    df.to_csv(name + "_avr.csv")
    Graph_avr_tc_tl(name + "_avr.csv").savefig(name + "_avr.png")
    #vio box
    tr_trs_repeat = np.repeat(np.arange(trial),generation)
    ge_ges_repeat = np.tile(ge_ges, trial)
    df = pd.DataFrame({"tr":tr_trs_repeat, "ge":ge_ges_repeat, "tc":tc_all_ges_trs, "tl":tl_all_ges_trs, "ln":ln_all_ges_trs})
    df.to_csv(name + "_all.csv")
    df = Graph_all_tc_tl_dfexplode(name + "_all.csv")
    # time2 = time.time()
    # Graph_all_vio(df, ylabel="tc").savefig(name + "_all_vio_tc.png")
    # time3 = time.time()
    # print("vio"+Elapsed_time_hms(time3-time2))
    # Graph_all_vio(df, ylabel="tl").savefig(name + "_all_vio_tl.png")
    # Graph_all_vio(df, ylabel="ln").savefig(name + "_all_vio_ln.png")
    time4 = time.time()
    Graph_all_box(df, ylabel="tc").savefig(name + "_all_box_tc.png")
    time5 = time.time()
    print("box"+Elapsed_time_hms(time5-time4))
    Graph_all_box(df, ylabel="tl").savefig(name + "_all_box_tl.png")
    Graph_all_box(df, ylabel="ln").savefig(name + "_all_box_ln.png")
    df = pd.DataFrame({"ge":ge_ges, "linkmatrix":linkmatrix_ges_tr0})
    df.to_csv(name + "_tr0_network.csv")
    time6 = time.time()
    Graph_network_ani(linkmatrix_ges=linkmatrix_ges_tr0).save(name + "_tr0_network.gif", writer='pillow', fps=60)
    time7 = time.time()
    print("ani"+Elapsed_time_hms(time7-time6))
    print("all"+Elapsed_time_hms(time7-time0))
#start_le_full()

def start_fo_null(): #ok based
    name = inspect.currentframe().f_code.co_name
    tc_avr_ges_trs,tf_avr_ges_trs,ln_avr_ges_trs = [],[],[]
    tc_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs = [],[],[]
    linkmatrix_ges_tr0 = []
    for tr in range(trial):
        tc = Initialize_value()
        tf = Initialize_value()
        linkmatrix = Initialize_linkmatrix_null()
        tc_avr_ges,tf_avr_ges,ln_avr_ges = [],[],[]
        tc_all_ges,tf_all_ges,ln_all_ges = [],[],[]
        for ge in range(generation):
            for ro in range(roound):
                if ro == 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #追加,今回のリンク数を調べる
                    coop_ro = Coop_ro_zero(tc) #これだけで自分の協力非協力決める
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #変更
                    count_game_ge = np.where(lnum_ro>0, 1, 0) #変更
                    count_coop_game_ge = np.where((lnum_ro>0)&(coop_ro==1), 1, 0) #変更
                    count_poff_ge = poff_ro
                if ro > 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #今回のリンク数調べる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更前の協力数調べる
                    coop_ro = Coop_ro_nonzero(cnum_ro=cnum_ro,lnum_ro=lnum_ro, tc=tc) #それで自分の協力非協力きめる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #利得求める
                    count_game_ge += np.where(lnum_ro>0, 1, 0)
                    count_coop_game_ge += np.where((lnum_ro>0)&(coop_ro==1), 1, 0)
                    count_poff_ge += poff_ro
                if ro < roound-1:
                    coop_ratio = np.divide(count_coop_game_ge, count_game_ge, where=count_game_ge>0)
                    linkmatrix = Leave_Form_tf(linkmatrix=linkmatrix,coop_ratio=coop_ratio, tf=tf)
            print(str(tr)+"tr-"+str(ge)+"ge")
            ln = np.sum(linkmatrix,axis=1)
            #sellection
            m_random = Randomn()
            cho = Linked_choice(linkmatrix, count_poff_ge, cho=[])
            tc,tf = Selection_tc_tf(m_random=m_random, count_poff_ge=count_poff_ge, cho=cho, tc_pre=tc, tf_pre=tf)
            #mutation
            tc,tf = Mutation_tc_tf(m_random=m_random,tc_pre=tc,tf_pre=tf)
            #graph
            tc_avr_ges.append(mean(tc)) #ok
            tf_avr_ges.append(mean(tf))
            ln_avr_ges.append(mean(ln))
            tc_all_ges.append(tc)#new
            tf_all_ges.append(tf)#new
            ln_all_ges.append(ln)#new
            if tr == 0:
                linkmatrix_ges_tr0.append(linkmatrix)
        tc_avr_ges_trs.append(tc_avr_ges) #ok
        tf_avr_ges_trs.append(tf_avr_ges)
        ln_avr_ges_trs.append(ln_avr_ges)
        tc_all_ges_trs.extend(tc_all_ges)
        tf_all_ges_trs.extend(tf_all_ges)
        ln_all_ges_trs.extend(ln_all_ges)
    time1 = time.time()#new
    print("fin" + Elapsed_time_hms(elapsed_time=(time1-time0)))#new
    #oresen
    ge_ges = np.arange(generation)
    ln_avr_ges_trs_avr = np.mean(ln_avr_ges_trs, axis=0)
    tc_avr_ges_trs_avr = np.mean(tc_avr_ges_trs, axis=0)
    tf_avr_ges_trs_avr = np.mean(tf_avr_ges_trs, axis=0)
    df = pd.DataFrame({"ge":ge_ges,"tc":tc_avr_ges_trs_avr,"tf":tf_avr_ges_trs_avr, "ln":ln_avr_ges_trs_avr})
    df.to_csv(name + "_avr.csv")
    Graph_avr_tc_tf(name + "_avr.csv").savefig(name + "_avr.png")
    #vio box
    tr_trs_repeat = np.repeat(np.arange(trial),generation)
    ge_ges_repeat = np.tile(ge_ges, trial)
    df = pd.DataFrame({"tr":tr_trs_repeat, "ge":ge_ges_repeat, "tc":tc_all_ges_trs, "tf":tf_all_ges_trs, "ln":ln_all_ges_trs})
    df.to_csv(name + "_all.csv")
    df = Graph_all_tc_tf_dfexplode(name + "_all.csv")
    # time2 = time.time()
    # Graph_all_vio(df, ylabel="tc").savefig(name + "_all_vio_tc.png")
    # time3 = time.time()
    # print("vio"+Elapsed_time_hms(time3-time2))
    # Graph_all_vio(df, ylabel="tf").savefig(name + "_all_vio_tf.png")
    # Graph_all_vio(df, ylabel="ln").savefig(name + "_all_vio_ln.png")
    time4 = time.time()
    Graph_all_box(df, ylabel="tc").savefig(name + "_all_box_tc.png")
    time5 = time.time()
    print("box"+Elapsed_time_hms(time5-time4))
    Graph_all_box(df, ylabel="tf").savefig(name + "_all_box_tf.png")
    Graph_all_box(df, ylabel="ln").savefig(name + "_all_box_ln.png")
    df = pd.DataFrame({"ge":ge_ges, "linkmatrix":linkmatrix_ges_tr0})
    df.to_csv(name + "_tr0_network.csv")
    time6 = time.time()
    Graph_network_ani(linkmatrix_ges=linkmatrix_ges_tr0).save(name + "_tr0_network.gif", writer='pillow', fps=1)
    time7 = time.time()
    print("ani"+Elapsed_time_hms(time7-time6))
    print("all"+Elapsed_time_hms(time7-time0))
#start_fo_null()


#1122-
def Initialize_linkmatrix_ba(): #完成
    ba = nx.barabasi_albert_graph(n,bam) #n=node number, m=new node's link number
    return nx.to_numpy_array(ba, dtype=int)
def start_bo_ba():
    name = inspect.currentframe().f_code.co_name
    tc_avr_ges_trs,tl_avr_ges_trs,tf_avr_ges_trs,ln_avr_ges_trs = [],[],[],[]
    tc_all_ges_trs,tl_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs = [],[],[],[]
    linkmatrix_ges_tr0 = []
    for tr in range(trial):
        tc = Initialize_value()
        tl = Initialize_value()
        tf = Initialize_value()
        linkmatrix = Initialize_linkmatrix_ba()
        tc_avr_ges,tl_avr_ges,tf_avr_ges,ln_avr_ges = [],[],[],[]
        tc_all_ges,tl_all_ges,tf_all_ges,ln_all_ges = [],[],[],[]
        for ge in range(generation):
            for ro in range(roound):
                if ro == 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #追加,今回のリンク数を調べる
                    coop_ro = Coop_ro_zero(tc) #これだけで自分の協力非協力決める
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #変更
                    count_game_ge = np.where(lnum_ro>0, 1, 0) #変更
                    count_coop_game_ge = np.where((lnum_ro>0)&(coop_ro==1), 1, 0) #変更
                    count_poff_ge = poff_ro
                if ro > 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #今回のリンク数調べる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更前の協力数調べる
                    coop_ro = Coop_ro_nonzero(cnum_ro=cnum_ro,lnum_ro=lnum_ro, tc=tc) #それで自分の協力非協力きめる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #利得求める
                    count_game_ge += np.where(lnum_ro>0, 1, 0)
                    count_coop_game_ge += np.where((lnum_ro>0)&(coop_ro==1), 1, 0)
                    count_poff_ge += poff_ro
                if ro < roound-1:
                    coop_ratio = np.divide(count_coop_game_ge, count_game_ge, where=count_game_ge>0)
                    linkmatrix = Leave_Form_tl_tf(linkmatrix=linkmatrix,coop_ratio=coop_ratio, tl=tl, tf=tf)
            print(str(tr)+"tr-"+str(ge)+"ge")
            ln = np.sum(linkmatrix,axis=1)
            #sellection
            m_random = Randomn()
            cho = Linked_choice(linkmatrix, count_poff_ge, cho=[])
            tc,tl,tf = Selection_tc_tl_tf(m_random=m_random,count_poff_ge=count_poff_ge,cho=cho,tc_pre=tc,tl_pre=tl,tf_pre=tf)
            #mutation
            tc,tl,tf = Mutation_tc_tl_tf(m_random=m_random,tc_pre=tc,tl_pre=tl,tf_pre=tf)
            #graph
            tc_avr_ges.append(mean(tc)) #ok
            tl_avr_ges.append(mean(tl))
            tf_avr_ges.append(mean(tf))
            ln_avr_ges.append(mean(ln))
            tc_all_ges.append(tc)
            tl_all_ges.append(tl)
            tf_all_ges.append(tf)
            ln_all_ges.append(ln)
            if tr == 0:
                linkmatrix_ges_tr0.append(linkmatrix)
        tc_avr_ges_trs.append(tc_avr_ges) #ok
        tl_avr_ges_trs.append(tl_avr_ges)
        tf_avr_ges_trs.append(tf_avr_ges)
        ln_avr_ges_trs.append(ln_avr_ges)
        tc_all_ges_trs.extend(tc_all_ges)
        tl_all_ges_trs.extend(tl_all_ges)
        tf_all_ges_trs.extend(tf_all_ges)
        ln_all_ges_trs.extend(ln_all_ges)
    time1 = time.time()#new
    print("fin" + Elapsed_time_hms(elapsed_time=(time1-time0)))#new
    #oresen
    ge_ges = np.arange(generation)
    ln_avr_ges_trs_avr = np.mean(ln_avr_ges_trs, axis=0)
    tc_avr_ges_trs_avr = np.mean(tc_avr_ges_trs, axis=0)
    tl_avr_ges_trs_avr = np.mean(tl_avr_ges_trs, axis=0)
    tf_avr_ges_trs_avr = np.mean(tf_avr_ges_trs, axis=0)
    df = pd.DataFrame({"ge":ge_ges,"tc":tc_avr_ges_trs_avr,"tl":tl_avr_ges_trs_avr,"tf":tf_avr_ges_trs_avr,"ln":ln_avr_ges_trs_avr})
    df.to_csv(name + "_avr.csv")
    Graph_avr_tc_tl_tf(name + "_avr.csv").savefig(name + "_avr.png")
    #vio box
    tr_trs_repeat = np.repeat(np.arange(trial),generation)
    ge_ges_repeat = np.tile(ge_ges, trial)
    df = pd.DataFrame({"tr":tr_trs_repeat,"ge":ge_ges_repeat,"tc":tc_all_ges_trs,"tl":tl_all_ges_trs,"tf":tf_all_ges_trs,"ln":ln_all_ges_trs})
    df.to_csv(name + "_all.csv")
    df = Graph_all_tc_tl_tf_dfexplode(name + "_all.csv")
    # time2 = time.time()
    # Graph_all_vio(df, ylabel="tc").savefig(name + "_all_vio_tc.png")
    # time3 = time.time()
    # print("vio"+Elapsed_time_hms(time3-time2))
    # Graph_all_vio(df, ylabel="tl").savefig(name + "_all_vio_tl.png")
    # Graph_all_vio(df, ylabel="tf").savefig(name + "_all_vio_tf.png")
    # Graph_all_vio(df, ylabel="ln").savefig(name + "_all_vio_ln.png")
    time4 = time.time()
    Graph_all_box(df, ylabel="tc").savefig(name + "_all_box_tc.png")
    time5 = time.time()
    print("box"+Elapsed_time_hms(time5-time4))
    Graph_all_box(df, ylabel="tl").savefig(name + "_all_box_tl.png")
    Graph_all_box(df, ylabel="tf").savefig(name + "_all_box_tf.png")
    Graph_all_box(df, ylabel="ln").savefig(name + "_all_box_ln.png")
    df = pd.DataFrame({"ge":ge_ges, "linkmatrix":linkmatrix_ges_tr0})
    df.to_csv(name + "_tr0_network.csv")
    time6 = time.time()
    Graph_network_ani(linkmatrix_ges=linkmatrix_ges_tr0).save(name + "_tr0_network.gif", writer='pillow', fps=60)
    time7 = time.time()
    print("ani"+Elapsed_time_hms(time7-time6))
    print("all"+Elapsed_time_hms(time7-time0))
start_bo_ba()

def start_bo_null():
    name = inspect.currentframe().f_code.co_name
    tc_avr_ges_trs,tl_avr_ges_trs,tf_avr_ges_trs,ln_avr_ges_trs = [],[],[],[]
    tc_all_ges_trs,tl_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs = [],[],[],[]
    linkmatrix_ges_tr0 = []
    for tr in range(trial):
        tc = Initialize_value()
        tl = Initialize_value()
        tf = Initialize_value()
        linkmatrix = Initialize_linkmatrix_null()
        tc_avr_ges,tl_avr_ges,tf_avr_ges,ln_avr_ges = [],[],[],[]
        tc_all_ges,tl_all_ges,tf_all_ges,ln_all_ges = [],[],[],[]
        for ge in range(generation):
            for ro in range(roound):
                if ro == 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #追加,今回のリンク数を調べる
                    coop_ro = Coop_ro_zero(tc) #これだけで自分の協力非協力決める
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #変更
                    count_game_ge = np.where(lnum_ro>0, 1, 0) #変更
                    count_coop_game_ge = np.where((lnum_ro>0)&(coop_ro==1), 1, 0) #変更
                    count_poff_ge = poff_ro
                if ro > 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #今回のリンク数調べる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更前の協力数調べる
                    coop_ro = Coop_ro_nonzero(cnum_ro=cnum_ro,lnum_ro=lnum_ro, tc=tc) #それで自分の協力非協力きめる
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #利得求める
                    count_game_ge += np.where(lnum_ro>0, 1, 0)
                    count_coop_game_ge += np.where((lnum_ro>0)&(coop_ro==1), 1, 0)
                    count_poff_ge += poff_ro
                if ro < roound-1:
                    coop_ratio = np.divide(count_coop_game_ge, count_game_ge, where=count_game_ge>0)
                    linkmatrix = Leave_Form_tl_tf(linkmatrix=linkmatrix,coop_ratio=coop_ratio, tl=tl, tf=tf)
            print(str(tr)+"tr-"+str(ge)+"ge")
            ln = np.sum(linkmatrix,axis=1)
            #sellection
            m_random = Randomn()
            cho = Linked_choice(linkmatrix, count_poff_ge, cho=[])
            tc,tl,tf = Selection_tc_tl_tf(m_random=m_random,count_poff_ge=count_poff_ge,cho=cho,tc_pre=tc,tl_pre=tl,tf_pre=tf)
            #mutation
            tc,tl,tf = Mutation_tc_tl_tf(m_random=m_random,tc_pre=tc,tl_pre=tl,tf_pre=tf)
            #graph
            tc_avr_ges.append(mean(tc)) #ok
            tl_avr_ges.append(mean(tl))
            tf_avr_ges.append(mean(tf))
            ln_avr_ges.append(mean(ln))
            tc_all_ges.append(tc)
            tl_all_ges.append(tl)
            tf_all_ges.append(tf)
            ln_all_ges.append(ln)
            if tr == 0:
                linkmatrix_ges_tr0.append(linkmatrix)
        tc_avr_ges_trs.append(tc_avr_ges) #ok
        tl_avr_ges_trs.append(tl_avr_ges)
        tf_avr_ges_trs.append(tf_avr_ges)
        ln_avr_ges_trs.append(ln_avr_ges)
        tc_all_ges_trs.extend(tc_all_ges)
        tl_all_ges_trs.extend(tl_all_ges)
        tf_all_ges_trs.extend(tf_all_ges)
        ln_all_ges_trs.extend(ln_all_ges)
    time1 = time.time()#new
    print("fin" + Elapsed_time_hms(elapsed_time=(time1-time0)))#new
    #oresen
    ge_ges = np.arange(generation)
    ln_avr_ges_trs_avr = np.mean(ln_avr_ges_trs, axis=0)
    tc_avr_ges_trs_avr = np.mean(tc_avr_ges_trs, axis=0)
    tl_avr_ges_trs_avr = np.mean(tl_avr_ges_trs, axis=0)
    tf_avr_ges_trs_avr = np.mean(tf_avr_ges_trs, axis=0)
    df = pd.DataFrame({"ge":ge_ges,"tc":tc_avr_ges_trs_avr,"tl":tl_avr_ges_trs_avr,"tf":tf_avr_ges_trs_avr,"ln":ln_avr_ges_trs_avr})
    df.to_csv(name + "_avr.csv")
    Graph_avr_tc_tl_tf(name + "_avr.csv").savefig(name + "_avr.png")
    #vio box
    tr_trs_repeat = np.repeat(np.arange(trial),generation)
    ge_ges_repeat = np.tile(ge_ges, trial)
    df = pd.DataFrame({"tr":tr_trs_repeat,"ge":ge_ges_repeat,"tc":tc_all_ges_trs,"tl":tl_all_ges_trs,"tf":tf_all_ges_trs,"ln":ln_all_ges_trs})
    df.to_csv(name + "_all.csv")
    df = Graph_all_tc_tl_tf_dfexplode(name + "_all.csv")
    # time2 = time.time()
    # Graph_all_vio(df, ylabel="tc").savefig(name + "_all_vio_tc.png")
    # time3 = time.time()
    # print("vio"+Elapsed_time_hms(time3-time2))
    # Graph_all_vio(df, ylabel="tl").savefig(name + "_all_vio_tl.png")
    # Graph_all_vio(df, ylabel="tf").savefig(name + "_all_vio_tf.png")
    # Graph_all_vio(df, ylabel="ln").savefig(name + "_all_vio_ln.png")
    time4 = time.time()
    Graph_all_box(df, ylabel="tc").savefig(name + "_all_box_tc.png")
    time5 = time.time()
    print("box"+Elapsed_time_hms(time5-time4))
    Graph_all_box(df, ylabel="tl").savefig(name + "_all_box_tl.png")
    Graph_all_box(df, ylabel="tf").savefig(name + "_all_box_tf.png")
    Graph_all_box(df, ylabel="ln").savefig(name + "_all_box_ln.png")
    df = pd.DataFrame({"ge":ge_ges, "linkmatrix":linkmatrix_ges_tr0})
    df.to_csv(name + "_tr0_network.csv")
    time6 = time.time()
    Graph_network_ani(linkmatrix_ges=linkmatrix_ges_tr0).save(name + "_tr0_network.gif", writer='pillow', fps=60)
    time7 = time.time()
    print("ani"+Elapsed_time_hms(time7-time6))
    print("all"+Elapsed_time_hms(time7-time0))
start_bo_null()

###note
# np.array([0,0],[0,0])でリストからナンパイ
# tc,tl,tf = selection
# gameカウントはリンクが繋がっている回数
# count_coopは繋がっていない時のは排除する
# npの計算では全て初期値参考、変遷は考えない、最後に一括更新
#nullは五個でなか三つ→一つに減ったリンクマトリクスだけ
#listはextendでリストを解除して中身だけ追加
#df.explodeでリストをばらして縦のデータにできる
#plotlyはあとでやる
#1122 tc,tl,tfの初期値に偏りを持たせるを具体的にしよう。始めをスケールフリーネットワークにしたらどうなるんだろう。評価項目にスケールフリー性とかネットワークの特徴量追加する？利得の計算を前の協力非協力ではなく今回でやる
#1122 なんで切り貼りは最終ラウンドやらないんだっけ
#1122 前回貰った改善点を列挙しておこう
#1123 tlは0.5安定で、tfが暴れるのはtfは繋がっているから真似するセレクションの影響が強い、利得は最初しか変わらないから、そいつの初期値に依存する？
#全員0.5スタートやってみてもいいかも