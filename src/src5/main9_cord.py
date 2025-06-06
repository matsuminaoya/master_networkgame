#TODO:main7から大幅変更
#TODO:チョイスを修正、初期値揃えよう、フレキシブル化してみた、初期値は三つ別に選択にしたこれで論文書けるかも、リストappendではなくnumpyarryで最初に枠決めて高速化、他に高速化できそうなとこ探してるcalculate_cnum簡略化、linkedchoice高速化、リンクマトリクス書き換え逐次最適化切り貼り動作確認okここのミスはない、簡単に動作確認できる。グラフ作成時のコメント削除、毎ラウンド0で協力非協力が初期化されるのはおかしいから修正main2→main3、初期値をグラフに入れる、協力非協力グラフに入れる、cnumの助長を減らす、グラフクローズ,silver始動グラフ色変える,dnの計算ミスge減らさない、tctltf以外は遺伝しない！←ココ、gをずらす、間違いを探す
#30時間12分の処理→00時間42分14秒に→これは三つ分、三時間くらいだった切り貼り最適化できてないけど
#TODO:main5→main6 tcによる初回のCorDの決定を変更
#Tclエラーの対処法が課題→pythonのインストール時にtcl/tkにチェックしてるのにできない→pyのver下げたらいけるだろ→いけた。特にpathを通す必要とかはない。
#whiteはpy3.11.4 at windowns
#.\venv\Scripts\activate.ps1 仮想環境明示
#pip install は必ず仮想環境で、（venv）を必ず確認
#python.exe -m pip install --upgrade pip
#import gc
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

time0 = time.time()
nowdate = datetime.now().strftime("%Y%m%d-%H%M")

#grobal value
n = 100 #OK#100 #4
bam = 2 #OK#banetwork'S new node's link number
a = 1.0 
bene = 2.0
cost = 1.0
mutation = 0.01
#tr>ge>ro>work
#trial = 1 #1,10 #2
#generation = 10000 #5001 #3 #mod(ani_step)=1ないとエラーが発生するので注意 #初期値が0に入ったから5000でもok5001になる→なくなった
#roound = 100 #100
#work = 5000 #5000
#graph_step
g_step = 100 #100
ani_step = 100 #100

#define
def Randomn(): #ok
    rng = np.random.default_rng()
    return rng.random(n)

def Sigmoid(x): #ok
    return np.exp(np.minimum(x,0)) / (1 + np.exp(-np.abs(x)))

def Initialize_value_random(): #ok#名称変更
    rng = np.random.default_rng()
    return 0.1*rng.integers(12,size=(1,n))[0]
def Initialize_value_zero(): #ok1130
    return np.zeros(n)
def Initialize_value_eleven():#ok1130
    return np.full(n, 1.1)


#name = "Initialize_value_eleven"
#print(eval(name)())

def Initialize_linkmatrix_full(): #ok
    linkmatrix_rvs = np.identity(n)
    return np.where(linkmatrix_rvs==0,1,0)
def Initialize_linkmatrix_null(): #ok
    return np.zeros((n,n),dtype=np.int16)
def Initialize_linkmatrix_ba(): #完成
    ba = nx.barabasi_albert_graph(n,bam) #n=node number, m=new node's link number
    return nx.to_numpy_array(ba, dtype=int)

def Calculate_cnum(coop_ro, linkmatrix): #ok#TODO:簡略化#pok
    # coop_ro_rvs = np.where(coop_ro==0,1,0) #ok
    # noncoop_index_ro = np.nonzero(coop_ro_rvs) #ok ゼロでないところしかピックできない
    # linkmatrix_del = np.delete(linkmatrix, noncoop_index_ro, 1) #非協力者の列削除
    # cnum_ro = np.sum(linkmatrix_del,axis=1)
    noncoop_index_ro = np.where(coop_ro == 0)[0]  # 非協力者のインデックス
    return np.sum(np.delete(linkmatrix, noncoop_index_ro, axis=1), axis=1) # 非協力者の列を削除し、残りのリンク数を計算

def Calculate_poff_ro(coop_ro, lnum_ro, cnum_ro): #ok #利得は人数で割ってる
    poff_ro_nodiv = np.where(coop_ro==1, (cnum_ro*bene)-(lnum_ro*cost), cnum_ro*bene)
    poff_ro = np.divide(poff_ro_nodiv, lnum_ro, out=poff_ro_nodiv, where=(lnum_ro!=0))
    return poff_ro

# def Initialize_poff_full(coop_ro, cnum_ro): #ok
#     return np.where(coop_ro==1, cnum_ro*bene-(n-1)*cost, cnum_ro*bene)
# def Initialize_poff_null(): #ok
#     return np.zeros((1,n))[0]

# def Initialize_count_game_ge_full(): #ok
#     return np.ones((1,n))[0]
# def Initialize_count_game_ge_null(): #ok
#     return np.zeros((1,n))[0]

def Linked_choice(n, linkmatrix): #ok #cho=[] #TODO:高速化考案#pok
    #linked_poff = linkmatrix*count_poff_ge #ok # TODO:利得最大の人の真似をするのが残ってた、同じ人選ばれたらやらないだけでいいんじゃない
    # for k in range(n): 
    #     linked_cho = linkmatrix[k].nonzero()
    #     if len(linked_cho[0]) == 0: #自分をついか
    #         cho.append(k)
    #     else:
    #         cho_index = random.choice(linked_cho[0])
    #         cho.append(cho_index.item())
    # return cho
    
    # 行ごとに1のインデックスを取得
    ones_mask = (linkmatrix == 1) #array([[False,  True, False],[ True, False,  True],[False,  True, False]]) #Trueと1は同値、こっちから変えんでもよい
    # 各行インデックスを取得
    row_indices = np.arange(n) #0123...
    # 各行で1がある列のインデックスを選択
    cho = np.array([np.random.choice(np.nonzero(ones_mask[row])[0]) #もし [0] を付けない場合、タプル全体（(array([1, 3]),)）が返されます。
                               if np.any(ones_mask[row]) #少なくとも1つのTrue（非ゼロ）の要素が含まれているかどうかをチェック
                               else row 
                               for row in row_indices])
    return cho


def Selection_tc_tl_tf(m_random, count_poff_ge, cho, tc_pre, tl_pre, tf_pre): #okok
    fermi = Sigmoid(1.0*a*(count_poff_ge[cho]-count_poff_ge)) #TODO:引き算の大きさが影響してる？フェルミの傾き
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
    tc_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tc_pre<=1.0)), tc_pre+0.1, tc_pre)
    tc = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tc_pre>=0.1)), tc_pre-0.1, tc_1)
    plus_minus = Randomn()
    tl_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tl_pre<=1.0)), tl_pre+0.1, tl_pre)
    tl = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tl_pre>=0.1)), tl_pre-0.1, tl_1)
    plus_minus = Randomn()
    tf_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tf_pre<=1.0)), tf_pre+0.1, tf_pre)
    tf = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tf_pre>=0.1)), tf_pre-0.1, tf_1)
    return tc,tl,tf
def Mutation_tc_tl(m_random, tc_pre, tl_pre): #okok
    plus_minus = Randomn()
    tc_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tc_pre<=1.0)), tc_pre+0.1, tc_pre)
    tc = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tc_pre>=0.1)), tc_pre-0.1, tc_1)
    plus_minus = Randomn()
    tl_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tl_pre<=1.0)), tl_pre+0.1, tl_pre)
    tl = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tl_pre>=0.1)), tl_pre-0.1, tl_1)
    return tc,tl
def Mutation_tc_tf(m_random, tc_pre, tf_pre): #okok based
    plus_minus = Randomn()
    tc_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tc_pre<=1.0)), tc_pre+0.1, tc_pre)
    tc = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tc_pre>=0.1)), tc_pre-0.1, tc_1)
    plus_minus = Randomn()
    tf_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tf_pre<=1.0)), tf_pre+0.1, tf_pre)
    tf = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tf_pre>=0.1)), tf_pre-0.1, tf_1)
    return tc,tf

def Coop_ro_zero(tc): #ok
    tc_random = Randomn()
    return np.where(((tc/1.1)<=tc_random),1,0)#TODO:変更
def Coop_ro_nonzero(cnum_ro, lnum_ro, tc): #okok
    tc_random = Randomn()
    ratio_clink = np.divide(cnum_ro, lnum_ro, where=lnum_ro>0)
    coop_ro_1 = np.where(((lnum_ro>0)&(tc<=ratio_clink)), 1, 0)
    coop_ro_2 = np.where(((lnum_ro==0)&(tc<=tc_random)), 1, coop_ro_1)
    coop_ro = np.where(((lnum_ro==0)&(tc_random<tc)), 0, coop_ro_2)
    return coop_ro

def Leave_Form_tl_tf(n, work, linkmatrix, coop_ratio, tl, tf): #TODO:書き換え#workを明示#TODO: きりはりだとｔｃがさがる、ｇが少ないと協力的、ｇが100000、500、100・・・900,gが少ないとfullだと非協力的に、平均と累積べつべつ、
    # rng = np.random.default_rng()
    # pair_index = np.triu_indices(n, k=1)
    # x = rng.integers(((n-1)*n/2),size=(1,work))[0]
    # i = (pair_index[0][x])
    # j = (pair_index[1][x])#TODO:書き換えてみる、逐次的な操作ではmuskは逆効果らしい、単一のインデックスを使うほうが高速だって
    
    # mask_link_0 = linkmatrix[i, j] == 0# 現在リンクがないペアのインデックス
    # mask_link_1 = linkmatrix[i, j] == 1# 現在リンクがあるペアのインデックス
    # # リンク形成：coop_ratio[i] >= tf[j] and coop_ratio[j] >= tf[i] の場合
    # mask_form = (coop_ratio[i] >= tf[j]) & (coop_ratio[j] >= tf[i])
    # linkmatrix[i[mask_link_0 & mask_form], j[mask_link_0 & mask_form]] = 1
    # linkmatrix[j[mask_link_0 & mask_form], i[mask_link_0 & mask_form]] = 1
    # # リンク解除：coop_ratio[i] < tl[j] or coop_ratio[j] < tl[i] の場合
    # mask_break = (coop_ratio[i] < tl[j]) | (coop_ratio[j] < tl[i])
    # linkmatrix[i[mask_link_1 & mask_break], j[mask_link_1 & mask_break]] = 0
    # linkmatrix[j[mask_link_1 & mask_break], i[mask_link_1 & mask_break]] = 0
    # return linkmatrix

    # ランダムなペアを生成
    i = np.random.randint(0, n, size=work)
    j = np.random.randint(0, n, size=work)
    # 同じノード同士のペアを除外
    mask_diff = i != j
    i = i[mask_diff]
    j = j[mask_diff]
    # print(i)
    # print(j)
    # 事前にcoop_ratioを配列で取得しておく
    coop_ratio_i = coop_ratio[i]
    coop_ratio_j = coop_ratio[j]
    # ペアごとにリンク形成・解除を逐次処理
    for k in range(len(i)):
        cratio_i, cratio_j = coop_ratio_i[k], coop_ratio_j[k]
        tf_i, tf_j = tf[i[k]], tf[j[k]]
        tl_i, tl_j = tl[i[k]], tl[j[k]]
        # print(f"ペアは（{i[k]} , {j[k]}）")
        # リンクがない場合、リンクを作る
        if linkmatrix[i[k], j[k]] == 0:
            if (cratio_i >= tf_j) and (cratio_j >= tf_i):
                linkmatrix[i[k], j[k]] = 1
                linkmatrix[j[k], i[k]] = 1
                # print(f"リンク接続: {i[k]} と {j[k]} のリンクを接続")
        # リンクがある場合、リンクを解除する
        elif linkmatrix[i[k], j[k]] == 1:
            if (cratio_i < tl_j) or (cratio_j < tl_i):
                linkmatrix[i[k], j[k]] = 0
                linkmatrix[j[k], i[k]] = 0
                # print(f"リンク切断: {i[k]} と {j[k]} のリンクを切断")
    return linkmatrix

def Leave_Form_tl(n, work, linkmatrix, coop_ratio, tl): #ok#workを明示
    # rng = np.random.default_rng()
    # pair_index = np.triu_indices(n, k=1)
    # x = rng.integers(((n-1)*n/2),size=(1,work))[0]
    # i = (pair_index[0][x])
    # j = (pair_index[1][x])
    # pair = (i,j)
    # mask_l = ((linkmatrix[pair]==1) & ((coop_ratio[pair[0]]<tl[pair[1]]) | (coop_ratio[pair[1]]<tl[pair[0]])))
    # linkmatrix[pair[0][mask_l],pair[1][mask_l]] = 0
    # linkmatrix[pair[1][mask_l],pair[0][mask_l]] = 0
    # return linkmatrix
    # ランダムなペアを生成
    i = np.random.randint(0, n, size=work)
    j = np.random.randint(0, n, size=work)
    # 同じノード同士のペアを除外
    mask_diff = i != j
    i = i[mask_diff]
    j = j[mask_diff]
    # print(i)
    # print(j)
    # 事前にcoop_ratioを配列で取得しておく
    coop_ratio_i = coop_ratio[i]
    coop_ratio_j = coop_ratio[j]
    # ペアごとにリンク形成・解除を逐次処理
    for k in range(len(i)):
        cratio_i, cratio_j = coop_ratio_i[k], coop_ratio_j[k]
        #tf_i, tf_j = tf[i[k]], tf[j[k]]
        tl_i, tl_j = tl[i[k]], tl[j[k]]
        # print(f"ペアは（{i[k]} , {j[k]}）")
        # リンクがない場合、リンクを作る
        #if linkmatrix[i[k], j[k]] == 0:
            #if (cratio_i >= tf_j) and (cratio_j >= tf_i):
            #    linkmatrix[i[k], j[k]] = 1
            #    linkmatrix[j[k], i[k]] = 1
                # print(f"リンク接続: {i[k]} と {j[k]} のリンクを接続")
        # リンクがある場合、リンクを解除する
        if linkmatrix[i[k], j[k]] == 1:
            if (cratio_i < tl_j) or (cratio_j < tl_i):
                linkmatrix[i[k], j[k]] = 0
                linkmatrix[j[k], i[k]] = 0
                # print(f"リンク切断: {i[k]} と {j[k]} のリンクを切断")
    return linkmatrix
def Leave_Form_tf(n, work, linkmatrix, coop_ratio, tf): #ok#workを明示
    # rng = np.random.default_rng()
    # pair_index = np.triu_indices(n, k=1)
    # x = rng.integers(((n-1)*n/2),size=(1,work))[0]
    # i = (pair_index[0][x])
    # j = (pair_index[1][x])
    # pair = (i,j)
    # mask_f = ((linkmatrix[pair]==0) & ((coop_ratio[pair[0]]>=tf[pair[1]]) & (coop_ratio[pair[1]]>=tf[pair[0]])))
    # linkmatrix[pair[0][mask_f],pair[1][mask_f]] = 1
    # linkmatrix[pair[1][mask_f],pair[0][mask_f]] = 1
    # return linkmatrix
    # ランダムなペアを生成
    i = np.random.randint(0, n, size=work)
    j = np.random.randint(0, n, size=work)
    # 同じノード同士のペアを除外
    mask_diff = i != j
    i = i[mask_diff]
    j = j[mask_diff]
    # print(i)
    # print(j)
    # 事前にcoop_ratioを配列で取得しておく
    coop_ratio_i = coop_ratio[i]
    coop_ratio_j = coop_ratio[j]
    # ペアごとにリンク形成・解除を逐次処理
    for k in range(len(i)):
        cratio_i, cratio_j = coop_ratio_i[k], coop_ratio_j[k]
        tf_i, tf_j = tf[i[k]], tf[j[k]]
        #tl_i, tl_j = tl[i[k]], tl[j[k]]
        # print(f"ペアは（{i[k]} , {j[k]}）")
        # リンクがない場合、リンクを作る
        if linkmatrix[i[k], j[k]] == 0:
            if (cratio_i >= tf_j) and (cratio_j >= tf_i):
                linkmatrix[i[k], j[k]] = 1
                linkmatrix[j[k], i[k]] = 1
                # print(f"リンク接続: {i[k]} と {j[k]} のリンクを接続")
        # リンクがある場合、リンクを解除する
        # elif linkmatrix[i[k], j[k]] == 1:
        #     if (cratio_i < tl_j) or (cratio_j < tl_i):
        #         linkmatrix[i[k], j[k]] = 0
        #         linkmatrix[j[k], i[k]] = 0
                # print(f"リンク切断: {i[k]} と {j[k]} のリンクを切断")
    return linkmatrix

def Graph_avr_tc_tl_tf(csv, name): #
    df =pd.read_csv(csv)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["ge"],df["tc"],label="tc",color="tab:blue")
    ax1.plot(df["ge"],df["tl"],label="tl",color="tab:orange")
    ax1.plot(df["ge"],df["tf"],label="tf",color="tab:green")
    #ax1.plot(df["ge"],df["dn"],label="D",color="black",alpha=0.5) #TODO:追加
    ax1.set_ylim(0,1.1)
    ax1.set_xlabel("generation")
    ax2 = ax1.twinx()
    ax2.bar(df["ge"],df["ln"],color='lightblue',label="Link")
    ax2.set_ylim(0,100)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc="upper right")
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    #print(inspect.currentframe().f_code.co_name)
    plt.savefig(name+"/"+name+"_avr.png")
    plt.close()
    return plt
def Graph_avr_tc_tl(csv, name): #
    df =pd.read_csv(csv)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["ge"],df["tc"],label="tc",color="tab:blue")
    ax1.plot(df["ge"],df["tl"],label="tl",color="tab:orange")
    #ax1.plot(df["ge"],df["dn"],label="D",color="black",alpha=0.5) #TODO:追加
    ax1.set_ylim(0,1.1)
    ax1.set_xlabel("generation")
    ax2 = ax1.twinx()
    ax2.bar(df["ge"],df["ln"],color='lightblue',label="Link")
    ax2.set_ylim(0,100)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc="upper right")
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    #print(inspect.currentframe().f_code.co_name)
    plt.savefig(name+"/"+name+"_avr.png")
    plt.close()
    return plt
def Graph_avr_tc_tf(csv, name): #
    df = pd.read_csv(csv)
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot()
    ax1.plot(df["ge"],df["tc"],label="tc",color="tab:blue")
    ax1.plot(df["ge"],df["tf"],label="tf",color="tab:green")
    #ax1.plot(df["ge"],df["dn"],label="D",color="black",alpha=0.5) #TODO:追加
    ax1.set_ylim(-0.09,1.19)
    ax1.set_xlabel("generation")
    ax2 = ax1.twinx()
    ax2.bar(df["ge"],df["ln"],color='lightblue',label="Link")
    ax2.set_ylim(-0.09,100.09)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2 ,loc="upper right")
    ax1.set_zorder(1)
    ax2.set_zorder(0)
    ax1.patch.set_alpha(0)
    #print(inspect.currentframe().f_code.co_name)
    plt.savefig(name+"/"+name+"_avr.png")
    plt.close()
    return

def Graph_all_dfgstep(csv): #TODO:名称変更、機能変わったからね
    df = pd.read_csv(csv)
    df = df[df["ge"]%g_step==0]
    # df["tc"] = df["tc"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    # df["tl"] = df["tl"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    # df["tf"] = df["tf"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    # df["ln"] = df["ln"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    # df = df.explode(["tc", "tl", "tf", "ln"], ignore_index=True)
    # df["tc"] = df["tc"].astype(float)
    # df["tl"] = df["tl"].astype(float)
    # df["tf"] = df["tf"].astype(float)
    # df["ln"] = df["ln"].astype(float)
    #print(inspect.currentframe().f_code.co_name)
    return df
# def Graph_all_tc_tl_dfgstep(csv): #
#     df = pd.read_csv(csv)
#     df = df[df["ge"]%g_step==0] #TODフィルタリング
#     # df["tc"] = df["tc"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
#     # df["tl"] = df["tl"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
#     # df["ln"] = df["ln"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
#     # df = df.explode(["tc", "tl", "ln"], ignore_index=True)
#     # df["tc"] = df["tc"].astype(float)
#     # df["tl"] = df["tl"].astype(float) #TODリストをnpにしてたぽいけどもういらない
#     #print(inspect.currentframe().f_code.co_name)
#     return df
# def Graph_all_tc_tf_dfgstep(csv): #
#     df = pd.read_csv(csv)
#     df = df[df["ge"]%g_step==0]
#     # df["tc"] = df["tc"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
#     # df["tf"] = df["tf"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
#     # df["ln"] = df["ln"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
#     # df = df.explode(["tc", "tf", "ln"], ignore_index=True)
#     # df["tc"] = df["tc"].astype(float)
#     # df["tf"] = df["tf"].astype(float)
#     # df["ln"] = df["ln"].astype(float)
#     #print(inspect.currentframe().f_code.co_name)
#     return df

def Graph_all_vio(df, ylabel): #
    plt.figure(figsize=(20,10))
    plt.ylabel(ylabel)
    plt.xlabel("ge")
    sns.violinplot(x="ge",y=ylabel,data=df)
    #print(inspect.currentframe().f_code.co_name)
    return plt
def Graph_all_box(df, ylabel, name): #TODO:閉じるように変更
    plt.figure(figsize=(20,10))
    if ylabel != "ln":
        plt.ylim(-0.009,1.109)
    else:
        plt.ylim(-0.09,100.09)
    plt.ylabel(ylabel)
    plt.xlabel("ge")
    sns.boxplot(x="ge", y=ylabel, data=df)
    #print(inspect.currentframe().f_code.co_name)
    plt.savefig(name+"/"+name+"_all_box_"+ylabel+".png")
    plt.close()
    return
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
    #print(inspect.currentframe().f_code.co_name)
    return ani

def Elapsed_time_hms(elapsed_time): #ok
    time_h = int(elapsed_time / 3600)
    time_m = int(elapsed_time % 3600 / 60)
    time_s = int(elapsed_time % 60)
    return ":time: {0:02}:{1:02}:{2:02}".format(time_h, time_m, time_s)

def Plotly_network_ani(linkmatrix_ges): #future work
    korekarayaru = 1
    return

#TODO:for def testing
# print(
# Leave_Form(
#coop_ro=np.array([0,0,0])
# #cnum_ro=np.array([2,2,2])
# n = 3
# work = 6
# linkmatrix=np.array([[0,1,1],
#                      [1,0,1],
#                      [1,1,0]])
# #count_poff_ge=np.array([4,9,1]),
# #cho=[1,2,0],
# #tc=np.array([0.0,0.1,0.2]),
# tl=np.array([0.8,0.0,0.8])
# tf=np.array([0.1,0.9,0.1])
# coop_ratio = np.array([0.5,0.5,0.5]) #繋がれるのは0and2のみ、切れるのは0or2が相手のとき
# )
# )
# print(Leave_Form_tl_tf(n=n, work=work, linkmatrix=linkmatrix, coop_ratio=coop_ratio, tl=tl, tf=tf))
# print("stop here in debug")

# [000][000][000],tl=np.array([0.8,0.8,0.8]),tf=np.array([0.1,0.1,0.1]),coop_ratio = np.array([0.5,0.5,0.5]) #絶対繋がるし絶対切れる
# [2 0 1]
# [0 2 0]
# ペアは（2 , 0）
# リンク接続: 2 と 0 のリンクを接続
# ペアは（0 , 2）
# リンク切断: 0 と 2 のリンクを切断
# ペアは（1 , 0）
# リンク接続: 1 と 0 のリンクを接続
# [[0 1 0]
#  [1 0 0]
#  [0 0 0]]

# makeed 150116-
def start2(netresetges = "yorn", isocount1st = "yorn", isocount2to = "yorn", lorf = "lorf", ininet = "ininet", tcinivalue = "tcinivalue", tlinivalue = "tcinivalue", tfinivalue = "tcinivalue", trial = 0, generation = 0, roound = 0, work = 0, nowdate=nowdate):
    print("start"+" "+netresetges+" "+isocount1st+" "+isocount2to+" "+lorf+" "+ininet+" "+tcinivalue+" "+tlinivalue+" "+tfinivalue+" t"+str(trial)+" g"+str(generation)+" w"+str(work)+" :n="+str(n))#名前変更
    #name = "t"+str(trial)+"_w"+str(work)+"_" + lorf + "_"+ininet+"_"+tcinivalue+tlinivalue+tfinivalue+"_"+nowdate #フレキシブル名称変更これはファイル名になる
    name = "t"+str(trial)+"g"+str(generation)+"r"+str(roound)+"_w"+str(work)+"_"+lorf+"_"+ininet+"_"+netresetges+isocount1st+isocount2to+"_"+nowdate
    os.makedirs(name, exist_ok=False) #ifFalseフォルダ作成、同じ名前があるとエラー
    #make tr[] for stack data
    if lorf == "leave":
        # tc_avr_ges_trs,tl_avr_ges_trs,ln_avr_ges_trs, tc_all_ges_trs,tl_all_ges_trs,ln_all_ges_trs, linkmatrix_ges_tr0 =  np.empty((trial,generation)),np.empty((trial,generation)),np.empty((trial,generation)), np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)), np.empty(1)#一行に変更リストの中にリストを外して入れるextendをnpでやるnp.concatenate()を使うためにゼロで埋めない→vstackにしたから、npzeroでもいけるはず、あとでやる
        tc_all_ges_trs,tl_all_ges_trs,ln_all_ges_trs,cd_all_ges_trs = np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial))#一行に変更リストの中にリストを外して入れるextendをnpでやるnp.concatenate()を使うためにゼロで埋めない→vstackにしたから、npzeroでもいけるはず、あとでやる
    elif lorf == "form":
        # tc_avr_ges_trs,tf_avr_ges_trs,ln_avr_ges_trs, tc_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs, linkmatrix_ges_tr0 =  np.empty((trial,generation)),np.empty((trial,generation)),np.empty((trial,generation)), np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)), np.empty(1)
        tc_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs,cd_all_ges_trs = np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial))
    elif lorf == "both":
        # tc_avr_ges_trs,tl_avr_ges_trs,tf_avr_ges_trs,ln_avr_ges_trs, tc_all_ges_trs,tl_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs, linkmatrix_ges_tr0 = np.empty((trial,generation)),np.empty((trial,generation)),np.empty((trial,generation)),np.empty((trial,generation)), np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)), np.empty(1) #TODO:linkmatrix_ges_tr0は後で考える
        tc_all_ges_trs,tl_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs,cd_all_ges_trs = np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)) #TODO:linkmatrix_ges_tr0は後で考える
    for tr in range(trial):
        #make ge[] for stack data
        if lorf == "leave":
            # tc_avr_ges,tl_avr_ges,ln_avr_ges,dn_avr_ges, tc_all_ges,tl_all_ges,ln_all_ges,dn_all_ges = np.empty(generation),np.empty(generation),np.empty(generation),np.empty(generation), np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n))#一行に変更,=[]じゃダメ #リストappendではなく、npでゼロの場所確保してgeで格納、一つに100個入るなら2Dにしなきゃだめ、emptyのが早いらしい
            tc_all_ges,tl_all_ges,ln_all_ges,cd_all_ges = np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n))#一行に変更,=[]じゃダメ #リストappendではなく、npでゼロの場所確保してgeで格納、一つに100個入るなら2Dにしなきゃだめ、emptyのが早いらしい
        elif lorf == "form":
            # tc_avr_ges,tf_avr_ges,ln_avr_ges,dn_avr_ges, tc_all_ges,tf_all_ges,ln_all_ges,dn_all_ges = np.empty(generation),np.empty(generation),np.empty(generation),np.empty(generation), np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n))
            tc_all_ges,tf_all_ges,ln_all_ges,cd_all_ges = np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n))
        elif lorf == "both":
            # tc_avr_ges,tl_avr_ges,tf_avr_ges,ln_avr_ges,dn_avr_ges, tc_all_ges,tl_all_ges,tf_all_ges,ln_all_ges,dn_all_ges = np.empty(generation),np.empty(generation),np.empty(generation),np.empty(generation),np.empty(generation), np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n))
            tc_all_ges,tl_all_ges,tf_all_ges,ln_all_ges,cd_all_ges = np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n))
        #Initialize_values
        if lorf == "leave":
            tc = eval("Initialize_value_"+tcinivalue)()
            tl = eval("Initialize_value_"+tlinivalue)()#フレキシブル化#pok
        elif lorf == "form":
            tc = eval("Initialize_value_"+tcinivalue)()
            tf = eval("Initialize_value_"+tfinivalue)()
        elif lorf == "both":
            tc = eval("Initialize_value_"+tcinivalue)()
            tl = eval("Initialize_value_"+tlinivalue)()
            tf = eval("Initialize_value_"+tfinivalue)()
        if netresetges == "no":
            linkmatrix = eval("Initialize_linkmatrix_"+ininet)()#各geでリンクを初期化しない
        #ln = np.sum(linkmatrix,axis=1) #TODO:geごとにネットワークリセット
        
        for ge in range(generation):
            #初期値代入、グラフのgeは一つずつずれる0-4999→0初期値1-5000
            # if ge == 0:#TODO:最初の最初だけ
            #     tc_avr_ges[0] = np.mean(tc)#ok[np.float64(0.0), np.float64(0.025), np.float64(0.025)]ジェネレーション個
            #     ln_avr_ges[0] = np.mean(ln)#各geでの全員の平均リンクを入れていく
            #     tc_all_ges[0] = tc#[array([0., 0., 0., 0.]), array([0. , 0. , 0. , 0.1]), array([0. , 0. , 0.1, 0. ])]ジェネレーション行人数列
            #     ln_all_ges[0] = ln#各geでの全員のリンク数、1ge1234人目,2ge1234人目
            #     if lorf == "leave":
            #         tl_avr_ges[0] = np.mean(tl)
            #         tl_all_ges[0] = tl
            #     elif lorf == "form":
            #         tf_avr_ges[0] = np.mean(tf)
            #         tf_all_ges[0] = tf
            #     elif lorf == "both":
            #         tl_avr_ges[0] = np.mean(tl)
            #         tf_avr_ges[0] = np.mean(tf)
            #         tl_all_ges[0] = tl
            #         tf_all_ges[0] = tf
            if netresetges == "yes":
                linkmatrix = eval("Initialize_linkmatrix_"+ininet)()#各geでリンクを初期化
            for ro in range(roound):
                if ro == 0:#1122変更 #TODO:ge==0追加しないと、協力非協力が過去を参照しなくなる、ゲームの回数協力した回数利得のみを初期化
                    # lnum_ro = np.sum(linkmatrix,axis=1) #追加,今回のリンク数を調べる
                    # if ge == 0:#TODO:追加
                    #     coop_ro = Coop_ro_zero(tc) #これだけで自分の協力非協力決める、初回だけ
                    #     dn = 1 - coop_ro
                    #     # dn_avr_ges[0] = np.mean(dn)
                    #     # dn_all_ges[0] = dn
                    # else:
                    #     #cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更前の協力数調べる
                    #     coop_ro = Coop_ro_nonzero(cnum_ro=cnum_ro,lnum_ro=lnum_ro, tc=tc) #それで自分の協力非協力きめる
                    #     dn = 1 - coop_ro#TODO:追加
                    lnum_ro = np.sum(linkmatrix,axis=1) #リンク数を初期化
                    coop_ro = Coop_ro_zero(tc)#協力非協力を初期化
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #各geで協力者数を初期化
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #各geでpoff_ro初期化
                    if isocount1st == "no":
                        count_game_ge = np.where(lnum_ro>0, 1, 0) #geme_ge初期化
                        count_coop_game_ge = np.where((lnum_ro>0)&(coop_ro==1), 1, 0) #geme_coop_ge初期化
                    if isocount1st == "yes":
                        count_game_ge = np.ones(n)
                        count_coop_game_ge = np.where((coop_ro==1), 1, 0)
                    count_poff_ge = poff_ro #poff_ge初期化
                if ro > 0:#1122変更
                    lnum_ro = np.sum(linkmatrix,axis=1) #リンク数の更新
                    #cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更前の協力数調べる
                    coop_ro = Coop_ro_nonzero(cnum_ro=cnum_ro,lnum_ro=lnum_ro, tc=tc) #それで自分の協力非協力きめる
                    # dn = 1 - coop_ro#TODO:追加
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix) #変更後の協力数調べる
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro) #利得求める
                    if isocount2to == "no":
                        count_game_ge += np.where(lnum_ro>0, 1, 0)
                        count_coop_game_ge += np.where((lnum_ro>0)&(coop_ro==1), 1, 0)
                    if isocount2to == "yes":
                        count_game_ge += 1
                        count_coop_game_ge += np.where((coop_ro==1), 1, 0)
                    count_poff_ge += poff_ro
                if ro < roound-1:
                    coop_ratio = np.divide(count_coop_game_ge, count_game_ge, where=count_game_ge>0)#coopratioの更新、ネットワークの更新
                    if lorf == "leave":
                        linkmatrix = Leave_Form_tl(n=n, work=work,linkmatrix=linkmatrix,coop_ratio=coop_ratio, tl=tl)
                    elif lorf == "form":
                        linkmatrix = Leave_Form_tf(n=n, work=work,linkmatrix=linkmatrix,coop_ratio=coop_ratio, tf=tf)
                    elif lorf == "both":
                        linkmatrix = Leave_Form_tl_tf(n=n, work=work,linkmatrix=linkmatrix,coop_ratio=coop_ratio, tl=tl, tf=tf)
            print(str(tr)+"tr-"+str(ge)+"ge")
            ln = np.sum(linkmatrix,axis=1)
            cd = coop_ro
            #sellection and mutation
            m_random = Randomn()
            cho = Linked_choice(n=n, linkmatrix=linkmatrix)#TODO:ランダム化入れなければ速くなるけど→ランダムは必要一応速くした
            if lorf == "leave":
                tc,tl = Selection_tc_tl(m_random=m_random, count_poff_ge=count_poff_ge, cho=cho, tc_pre=tc, tl_pre=tl)
                tc,tl = Mutation_tc_tl(m_random=m_random,tc_pre=tc,tl_pre=tl)
            elif lorf == "form":
                tc,tf = Selection_tc_tf(m_random=m_random, count_poff_ge=count_poff_ge, cho=cho, tc_pre=tc, tf_pre=tf)
                tc,tf = Mutation_tc_tf(m_random=m_random,tc_pre=tc,tf_pre=tf)
            elif lorf == "both":
                tc,tl,tf = Selection_tc_tl_tf(m_random=m_random, count_poff_ge=count_poff_ge, cho=cho, tc_pre=tc, tl_pre=tl, tf_pre=tf)
                tc,tl,tf = Mutation_tc_tl_tf(m_random=m_random,tc_pre=tc,tl_pre=tl,tf_pre=tf)
            #graph
            # geにおいて、平均値を入れる/全部入れる
            # if lorf == "leave":
            #     tc_avr_ges.append(mean(tc)) #ok
            #     tl_avr_ges.append(mean(tl))
            #     ln_avr_ges.append(mean(ln))#各geでの全員の平均リンクを入れていく
            #     tc_all_ges.append(tc)
            #     tl_all_ges.append(tl)
            #     ln_all_ges.append(ln)#各geでの全員のリンク数、1ge1234人目,2ge1234人目
            # if lorf == "form":
            #     tc_avr_ges.append(mean(tc)) #ok
            #     tf_avr_ges.append(mean(tf))
            #     ln_avr_ges.append(mean(ln))#各geでの全員の平均リンクを入れていく
            #     tc_all_ges.append(tc)
            #     tf_all_ges.append(tf)
            #     ln_all_ges.append(ln)#各geでの全員のリンク数、1ge1234人目,2ge1234人目
            # if lorf == "both":
            #     tc_avr_ges.append(mean(tc)) #ok
            #     tl_avr_ges.append(mean(tl))
            #     tf_avr_ges.append(mean(tf))
            #     ln_avr_ges.append(mean(ln))#各geでの全員の平均リンクを入れていく
            #     tc_all_ges.append(tc)
            #     tl_all_ges.append(tl)
            #     tf_all_ges.append(tf)
            #     ln_all_ges.append(ln)#各geでの全員のリンク数、1ge1234人目,2ge1234人目
            if lorf == "leave":
                # tl_avr_ges[ge] = np.mean(tl)
                tl_all_ges[ge] = tl
            elif lorf == "form":
                # tf_avr_ges[ge] = np.mean(tf)
                tf_all_ges[ge] = tf
            elif lorf == "both":
                # tl_avr_ges[ge] = np.mean(tl)
                # tf_avr_ges[ge] = np.mean(tf)
                tl_all_ges[ge] = tl
                tf_all_ges[ge] = tf
            # tc_avr_ges[ge] = np.mean(tc)#ok[np.float64(0.0), np.float64(0.025), np.float64(0.025)]ジェネレーション個
            # ln_avr_ges[ge] = np.mean(ln)#各geでの全員の平均リンクを入れていく
            #dn_avr_ges[ge] = np.mean(dn)
            tc_all_ges[ge] = tc#[array([0., 0., 0., 0.]), array([0. , 0. , 0. , 0.1]), array([0. , 0. , 0.1, 0. ])]ジェネレーション行人数列
            ln_all_ges[ge] = ln
            cd_all_ges[ge] = cd
            #dn_all_ges[ge] = dn
            #if tr == 0:
            #    linkmatrix_ges_tr0.append(linkmatrix) #トライアル0の場合は全ての世代でのネットワークを保存
        #trにおいて、ためる/解除してためる
        if lorf == "leave":
            # tl_avr_ges_trs[tr] = tl_avr_ges
            tl_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tl_all_ges.reshape(-1)#[array([0., 0., 0., 0.]), array([0., 0., 0., 0.]), array([0., 0., 0., 0.]), array([0., 0., 0., 0.]), array([0. , 0. , 0. , 0.1]), array([0. , 0. , 0.1, 0. ])]
            # tc_all_ges_trs = np.vstack((tc_all_ges_trs, tc_all_ges)) #.ravelで配列をコピーせずに一次元の一つの配列にする。その前は、concateireみないなやつと,ravelみたいなやつの合わせてをやってた。
            # tl_all_ges_trs = np.vstack((tl_all_ges_trs, tl_all_ges))#[array([0., 0., 0., 0.]), array([0., 0., 0., 0.]), array([0., 0., 0., 0.]), array([0., 0., 0., 0.]), array([0. , 0. , 0. , 0.1]), array([0. , 0. , 0.1, 0. ])]
            # ln_all_ges_trs = np.vstack((ln_all_ges_trs, ln_all_ges))#1試行目の1ge1234人目,2ge1234人目,2試行目の...[]解除、トライアルではまとめないジェネレーションではまとめる
        elif lorf == "form":
            # tf_avr_ges_trs[tr] = tf_avr_ges
            tf_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tf_all_ges.reshape(-1)#一次元にしてから入れてるreshape(1,-1)→二次元になっちゃう
        elif lorf == "both":
            # tl_avr_ges_trs[tr] = tl_avr_ges
            # tf_avr_ges_trs[tr] = tf_avr_ges
            tl_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tl_all_ges.reshape(-1)
            tf_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tf_all_ges.reshape(-1)
        # tc_avr_ges_trs[tr] = tc_avr_ges #ok[[np.float64(0.0), np.float64(0.0), np.float64(0.0)], [np.float64(0.0), np.float64(0.025), np.float64(0.025)]]トライアル行ジェネレーション列
        # ln_avr_ges_trs[tr] = ln_avr_ges#[1試行目の各geでの全員の平均利得],[2試行目の...
        #dn_avr_ges_trs[tr] = dn_avr_ges
        tc_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tc_all_ges.reshape(-1) #.ravelで配列をコピーせずに一次元の一つの配列にする。
        ln_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = ln_all_ges.reshape(-1)#1試行目の1ge1234人目,2ge1234人目,2試行目の...[]解除、トライアルではまとめないジェネレーションではまとめる
        cd_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = cd_all_ges.reshape(-1)
        #dn_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = dn_all_ges.reshape(-1)
    # time1 = time.time()#new
    # print("sim"+Elapsed_time_hms(elapsed_time=(time1-time0)))#new
    #oresen
    # ge_ges = np.arange(generation)
    #各ラウンドでの全員の平均、の試行平均でdf
    # tc_avr_ges_trs_avr = np.mean(tc_avr_ges_trs, axis=0)#array([0.    , 0.0125, 0.0125])ジェネレーション個
    # ln_avr_ges_trs_avr = np.mean(ln_avr_ges_trs, axis=0)#各ラウンドでの全員の平気利得、の試行平均
    #dn_avr_ges_trs_avr = np.mean(dn_avr_ges_trs, axis=0)
    # if lorf == "leave":
    #     tl_avr_ges_trs_avr = np.mean(tl_avr_ges_trs, axis=0)
    #     df = pd.DataFrame({"ge":ge_ges,"tc":tc_avr_ges_trs_avr,"tl":tl_avr_ges_trs_avr,"ln":ln_avr_ges_trs_avr})#,"dn":dn_avr_ges_trs_avr
    # elif lorf == "form":
    #     tf_avr_ges_trs_avr = np.mean(tf_avr_ges_trs, axis=0)
    #     df = pd.DataFrame({"ge":ge_ges,"tc":tc_avr_ges_trs_avr,"tf":tf_avr_ges_trs_avr,"ln":ln_avr_ges_trs_avr})
    # elif lorf == "both":
    #     tl_avr_ges_trs_avr = np.mean(tl_avr_ges_trs, axis=0)
    #     tf_avr_ges_trs_avr = np.mean(tf_avr_ges_trs, axis=0)
    #     df = pd.DataFrame({"ge":ge_ges,"tc":tc_avr_ges_trs_avr,"tl":tl_avr_ges_trs_avr,"tf":tf_avr_ges_trs_avr,"ln":ln_avr_ges_trs_avr})
    # df.to_csv(name+"/"+name+"_avr.csv")#フォルダの中に格納
    # if lorf == "leave": 
    #     Graph_avr_tc_tl(name+"/"+name+"_avr.csv", name)
    # elif lorf == "form": 
    #     Graph_avr_tc_tf(name+"/"+name+"_avr.csv", name)
    # elif lorf == "both": 
    #     Graph_avr_tc_tl_tf(name+"/"+name+"_avr.csv", name)
    #vio box
    tr_trs_repeat = np.repeat(np.arange(trial),(generation)*n)#000000000....1111111
    #tr_trs_repeat = tr_trs_repeat_h.reshape(-1,1)#[0],[1],[2]...に変換-1は自動計算、1列指定
    ge_ges_n = np.repeat(np.arange(generation),n)#1ge1ge1ge...5000ge5000geを
    ge_ges_repeat = np.tile(ge_ges_n, trial) #ntr繰り返す
    #ge_ges_repeat = ge_ges_repeat_h.reshape(-1,1)
    #全員のge×tr全てでdf
    if lorf == "leave":
        df = pd.DataFrame({"tr":tr_trs_repeat, "ge":ge_ges_repeat, "tc":tc_all_ges_trs, "tl":tl_all_ges_trs, "ln":ln_all_ges_trs, "cd":cd_all_ges_trs})#一行にしなきゃだめみたい#, "dn":dn_all_ges_trs
    elif lorf == "form":
        df = pd.DataFrame({"tr":tr_trs_repeat, "ge":ge_ges_repeat, "tc":tc_all_ges_trs, "tf":tf_all_ges_trs, "ln":ln_all_ges_trs, "cd":cd_all_ges_trs})
    elif lorf == "both":
        df = pd.DataFrame({"tr":tr_trs_repeat, "ge":ge_ges_repeat, "tc":tc_all_ges_trs, "tl":tl_all_ges_trs, "tf":tf_all_ges_trs, "ln":ln_all_ges_trs, "cd":cd_all_ges_trs})
    df.to_csv(name+"/"+name+"_all.csv")#フォルダの中に格納
    #df = Graph_all_dfgstep(name+"/"+name+"_all.csv")#フォルダの中に格納#TODO:
    # time2 = time.time()
    # Graph_all_vio(df, ylabel="tc").savefig(name + "_all_vio_tc.png")
    # time3 = time.time()
    # print("vio"+Elapsed_time_hms(time3-time2))
    # Graph_all_vio(df, ylabel="tl").savefig(name + "_all_vio_tl.png")
    # Graph_all_vio(df, ylabel="ln").savefig(name + "_all_vio_ln.png")
    # time4 = time.time()
    #box graph
    # Graph_all_box(df, ylabel="tc", name=name)#TODO:
    # Graph_all_box(df, ylabel="ln", name=name)
    # Graph_all_box(df, ylabel="dn", name=name)
    # if lorf == "leave":
    #     Graph_all_box(df, ylabel="tl", name=name)
    # elif lorf == "form":
    #     Graph_all_box(df, ylabel="tf", name=name)
    # elif lorf == "both":
    #     Graph_all_box(df, ylabel="tl", name=name)
    #     Graph_all_box(df, ylabel="tf", name=name)
    #network gif
    #df = pd.DataFrame({"ge":ge_ges, "linkmatrix":linkmatrix_ges_tr0})
    #df.to_csv(name + "_tr0_network.csv")
    #time6 = time.time()
    #Graph_network_ani(linkmatrix_ges=linkmatrix_ges_tr0).save(name + "_tr0_network.gif", writer='pillow', fps=60)
    time7 = time.time()
    #print("ani"+Elapsed_time_hms(time7-time6))
    print("all"+Elapsed_time_hms(time7-time0))

#work=5000, zerozerozero, tr=1 wookに関しては質問
#start2(netresetges="no", isocount1st="no", isocount2to="no", lorf="leave", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="no", isocount1st="yes", isocount2to="no", lorf="leave", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="no", isocount1st="yes", isocount2to="yes", lorf="leave", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=10, work=5000)
#start2(netresetges="yes", isocount1st="no", isocount2to="no", lorf="leave", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="yes", isocount1st="yes", isocount2to="no", lorf="leave", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="yes", isocount1st="yes", isocount2to="yes", lorf="leave", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)

#start2(netresetges="no", isocount1st="no", isocount2to="no", lorf="both", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="no", isocount1st="yes", isocount2to="no", lorf="both", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="no", isocount1st="yes", isocount2to="yes", lorf="both", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=5, work=5000)
#start2(netresetges="yes", isocount1st="no", isocount2to="no", lorf="both", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="yes", isocount1st="yes", isocount2to="no", lorf="both", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="yes", isocount1st="yes", isocount2to="yes", lorf="both", ininet="full", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)

#start2(netresetges="no", isocount1st="no", isocount2to="no", lorf="form", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="no", isocount1st="yes", isocount2to="no", lorf="form", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="no", isocount1st="yes", isocount2to="yes", lorf="form", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=5, work=5000)
#start2(netresetges="yes", isocount1st="no", isocount2to="no", lorf="form", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="yes", isocount1st="yes", isocount2to="no", lorf="form", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)

#start2(netresetges="yes", isocount1st="yes", isocount2to="yes", lorf="form", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=10, generation=100000, roound=10, work=5000)
#start2(netresetges="yes", isocount1st="yes", isocount2to="yes", lorf="form", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=10, generation=10000, roound=100, work=5000)
start2(netresetges="yes", isocount1st="yes", isocount2to="yes", lorf="form", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=10, generation=1000, roound=1000, work=5000)
start2(netresetges="yes", isocount1st="yes", isocount2to="yes", lorf="form", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=10, generation=100, roound=10000, work=5000)

#start2(netresetges="no", isocount1st="no", isocount2to="no", lorf="both", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="no", isocount1st="yes", isocount2to="no", lorf="both", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="no", isocount1st="yes", isocount2to="yes", lorf="both", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=5, work=5000)
#start2(netresetges="yes", isocount1st="no", isocount2to="no", lorf="both", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="yes", isocount1st="yes", isocount2to="no", lorf="both", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)
#start2(netresetges="yes", isocount1st="yes", isocount2to="yes", lorf="both", ininet="null", tcinivalue="zero", tlinivalue="zero", tfinivalue="zero", trial=1, work=5000)