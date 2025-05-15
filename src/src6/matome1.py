import os
import time
import random
import inspect
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from statistics import mean

nowdate = datetime.now().strftime("%Y%m%d-%H%M")

n = 100
bam = 2
a = 1.0 
bene = 2.0
cost = 1.0
mutation = 0.01

def Randomn():
    rng = np.random.default_rng()
    return rng.random(n)

def Sigmoid(x):
    return np.exp(np.minimum(x,0)) / (1 + np.exp(-np.abs(x)))

def Initialize_value_random():
    rng = np.random.default_rng()
    return 0.1*rng.integers(12,size=(1,n))[0]
def Initialize_value_zero():
    return np.zeros(n)
def Initialize_value_eleven():
    return np.full(n, 1.1)

def Initialize_linkmatrix_full():
    linkmatrix_rvs = np.identity(n)
    return np.where(linkmatrix_rvs==0,1,0)
def Initialize_linkmatrix_null():
    return np.zeros((n,n),dtype=np.int16)
def Initialize_linkmatrix_ba():
    ba = nx.barabasi_albert_graph(n,bam)
    return nx.to_numpy_array(ba, dtype=int)

def Calculate_cnum(coop_ro, linkmatrix):
    noncoop_index_ro = np.where(coop_ro == 0)[0]
    return np.sum(np.delete(linkmatrix, noncoop_index_ro, axis=1), axis=1)

def Calculate_poff_ro(coop_ro, lnum_ro, cnum_ro): #ok #利得は人数で割ってる
    poff_ro_nodiv = np.where(coop_ro==1, (cnum_ro*bene)-(lnum_ro*cost), cnum_ro*bene)
    poff_ro = np.divide(poff_ro_nodiv, lnum_ro, out=poff_ro_nodiv, where=(lnum_ro!=0))
    return poff_ro

def Linked_choice(n, linkmatrix):
    ones_mask = (linkmatrix == 1)
    row_indices = np.arange(n)
    cho = np.array([np.random.choice(np.nonzero(ones_mask[row])[0])
                               if np.any(ones_mask[row])
                               else row 
                               for row in row_indices])
    return cho


def Selection_tc_tl_tf(m_random, count_poff_ge, cho, tc_pre, tl_pre, tf_pre):
    fermi = Sigmoid(1.0*a*(count_poff_ge[cho]-count_poff_ge))
    f_random = Randomn()
    tc = np.where(((mutation<=m_random)&(f_random<fermi)), tc_pre[cho], tc_pre)
    f_random = Randomn()
    tl = np.where(((mutation<=m_random)&(f_random<fermi)), tl_pre[cho], tl_pre)
    f_random = Randomn()
    tf = np.where(((mutation<=m_random)&(f_random<fermi)), tf_pre[cho], tf_pre)
    return tc,tl,tf
def Selection_tc_tl(m_random, count_poff_ge, cho, tc_pre, tl_pre):
    fermi = Sigmoid(1.0*a*(count_poff_ge[cho]-count_poff_ge))
    f_random = Randomn()
    tc = np.where(((mutation<=m_random)&(f_random<fermi)), tc_pre[cho], tc_pre)
    f_random = Randomn()
    tl = np.where(((mutation<=m_random)&(f_random<fermi)), tl_pre[cho], tl_pre)
    return tc,tl
def Selection_tc_tf(m_random, count_poff_ge, cho, tc_pre, tf_pre):
    fermi = Sigmoid(1.0*a*(count_poff_ge[cho]-count_poff_ge))
    f_random = Randomn()
    tc = np.where(((mutation<=m_random)&(f_random<fermi)), tc_pre[cho], tc_pre)
    f_random = Randomn()
    tf = np.where(((mutation<=m_random)&(f_random<fermi)), tf_pre[cho], tf_pre)
    return tc,tf

def Mutation_tc_tl_tf(m_random, tc_pre, tl_pre, tf_pre):
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
def Mutation_tc_tl(m_random, tc_pre, tl_pre):
    plus_minus = Randomn()
    tc_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tc_pre<=1.0)), tc_pre+0.1, tc_pre)
    tc = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tc_pre>=0.1)), tc_pre-0.1, tc_1)
    plus_minus = Randomn()
    tl_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tl_pre<=1.0)), tl_pre+0.1, tl_pre)
    tl = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tl_pre>=0.1)), tl_pre-0.1, tl_1)
    return tc,tl
def Mutation_tc_tf(m_random, tc_pre, tf_pre):
    plus_minus = Randomn()
    tc_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tc_pre<=1.0)), tc_pre+0.1, tc_pre)
    tc = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tc_pre>=0.1)), tc_pre-0.1, tc_1)
    plus_minus = Randomn()
    tf_1 = np.where((((m_random<mutation)&(plus_minus<0.5))&(tf_pre<=1.0)), tf_pre+0.1, tf_pre)
    tf = np.where((((m_random<mutation)&(plus_minus>=0.5))&(tf_pre>=0.1)), tf_pre-0.1, tf_1)
    return tc,tf

def Coop_ro_zero(tc):
    tc_random = Randomn()
    return np.where(((tc/1.1)<=tc_random),1,0)
def Coop_ro_nonzero(cnum_ro, lnum_ro, tc):
    tc_random = Randomn()
    ratio_clink = np.divide(cnum_ro, lnum_ro, where=lnum_ro>0)
    coop_ro_1 = np.where(((lnum_ro>0)&(tc<=ratio_clink)), 1, 0)
    coop_ro_2 = np.where(((lnum_ro==0)&(tc<=tc_random)), 1, coop_ro_1)
    coop_ro = np.where(((lnum_ro==0)&(tc_random<tc)), 0, coop_ro_2)
    return coop_ro

def Leave_Form_tl_tf(n, work, linkmatrix, coop_ratio, tl, tf):
    i = np.random.randint(0, n, size=work)
    j = np.random.randint(0, n, size=work)
    mask_diff = i != j
    i = i[mask_diff]
    j = j[mask_diff]
    coop_ratio_i = coop_ratio[i]
    coop_ratio_j = coop_ratio[j]
    for k in range(len(i)):
        cratio_i, cratio_j = coop_ratio_i[k], coop_ratio_j[k]
        tf_i, tf_j = tf[i[k]], tf[j[k]]
        tl_i, tl_j = tl[i[k]], tl[j[k]]
        if linkmatrix[i[k], j[k]] == 0:
            if (cratio_i >= tf_j) and (cratio_j >= tf_i):
                linkmatrix[i[k], j[k]] = 1
                linkmatrix[j[k], i[k]] = 1
        elif linkmatrix[i[k], j[k]] == 1:
            if (cratio_i < tl_j) or (cratio_j < tl_i):
                linkmatrix[i[k], j[k]] = 0
                linkmatrix[j[k], i[k]] = 0
    return linkmatrix

def Leave_Form_tl(n, work, linkmatrix, coop_ratio, tl):
    i = np.random.randint(0, n, size=work)
    j = np.random.randint(0, n, size=work)
    mask_diff = i != j
    i = i[mask_diff]
    j = j[mask_diff]
    coop_ratio_i = coop_ratio[i]
    coop_ratio_j = coop_ratio[j]
    for k in range(len(i)):
        cratio_i, cratio_j = coop_ratio_i[k], coop_ratio_j[k]
        tl_i, tl_j = tl[i[k]], tl[j[k]]
        if linkmatrix[i[k], j[k]] == 1:
            if (cratio_i < tl_j) or (cratio_j < tl_i):
                linkmatrix[i[k], j[k]] = 0
                linkmatrix[j[k], i[k]] = 0
    return linkmatrix
def Leave_Form_tf(n, work, linkmatrix, coop_ratio, tf):
    i = np.random.randint(0, n, size=work)
    j = np.random.randint(0, n, size=work)
    mask_diff = i != j
    i = i[mask_diff]
    j = j[mask_diff]
    coop_ratio_i = coop_ratio[i]
    coop_ratio_j = coop_ratio[j]
    for k in range(len(i)):
        cratio_i, cratio_j = coop_ratio_i[k], coop_ratio_j[k]
        tf_i, tf_j = tf[i[k]], tf[j[k]]
        if linkmatrix[i[k], j[k]] == 0:
            if (cratio_i >= tf_j) and (cratio_j >= tf_i):
                linkmatrix[i[k], j[k]] = 1
                linkmatrix[j[k], i[k]] = 1
    return linkmatrix


def start2(netresetges = "yorn", isocount1st = "yorn", isocount2to = "yorn", lorf = "lorf", ininet = "ininet", tcinivalue = "tcinivalue", tlinivalue = "tcinivalue", tfinivalue = "tcinivalue", trial = 0, generation = 0, roound = 0, work = 0, nowdate=nowdate):
    print("start"+" "+netresetges+" "+isocount1st+" "+isocount2to+" "+lorf+" "+ininet+" "+tcinivalue+" "+tlinivalue+" "+tfinivalue+" t"+str(trial)+" g"+str(generation)+" w"+str(work)+" :n="+str(n))
    name = "t"+str(trial)+"g"+str(generation)+"r"+str(roound)+"_w"+str(work)+"_"+lorf+"_"+ininet+"_"+netresetges+isocount1st+isocount2to+"_"+nowdate
    os.makedirs(name, exist_ok=False)
    if lorf == "leave":
        tc_all_ges_trs,tl_all_ges_trs,ln_all_ges_trs,cd_all_ges_trs = np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial))
    elif lorf == "form":
        tc_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs,cd_all_ges_trs = np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial))
    elif lorf == "both":
        tc_all_ges_trs,tl_all_ges_trs,tf_all_ges_trs,ln_all_ges_trs,cd_all_ges_trs = np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial)),np.empty((n*(generation)*trial))
    for tr in range(trial):
        if lorf == "leave":
            tc_all_ges,tl_all_ges,ln_all_ges,cd_all_ges = np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n))
        elif lorf == "form":
            tc_all_ges,tf_all_ges,ln_all_ges,cd_all_ges = np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n))
        elif lorf == "both":
            tc_all_ges,tl_all_ges,tf_all_ges,ln_all_ges,cd_all_ges = np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n)),np.empty((generation,n))
        if lorf == "leave":
            tc = eval("Initialize_value_"+tcinivalue)()
            tl = eval("Initialize_value_"+tlinivalue)()
        elif lorf == "form":
            tc = eval("Initialize_value_"+tcinivalue)()
            tf = eval("Initialize_value_"+tfinivalue)()
        elif lorf == "both":
            tc = eval("Initialize_value_"+tcinivalue)()
            tl = eval("Initialize_value_"+tlinivalue)()
            tf = eval("Initialize_value_"+tfinivalue)()
        if netresetges == "no":
            linkmatrix = eval("Initialize_linkmatrix_"+ininet)()
        for ge in range(generation):
            if netresetges == "yes":
                linkmatrix = eval("Initialize_linkmatrix_"+ininet)()
            for ro in range(roound):
                if ro == 0:
                    lnum_ro = np.sum(linkmatrix,axis=1)
                    coop_ro = Coop_ro_zero(tc)
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix)
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro)
                    if isocount1st == "no":
                        count_game_ge = np.where(lnum_ro>0, 1, 0)
                        count_coop_game_ge = np.where((lnum_ro>0)&(coop_ro==1), 1, 0)
                    if isocount1st == "yes":
                        count_game_ge = np.ones(n)
                        count_coop_game_ge = np.where((coop_ro==1), 1, 0)
                    count_poff_ge = poff_ro
                if ro > 0:
                    lnum_ro = np.sum(linkmatrix,axis=1)
                    coop_ro = Coop_ro_nonzero(cnum_ro=cnum_ro,lnum_ro=lnum_ro, tc=tc)
                    cnum_ro = Calculate_cnum(coop_ro=coop_ro,linkmatrix=linkmatrix)
                    poff_ro = Calculate_poff_ro(coop_ro=coop_ro,lnum_ro=lnum_ro,cnum_ro=cnum_ro)
                    if isocount2to == "no":
                        count_game_ge += np.where(lnum_ro>0, 1, 0)
                        count_coop_game_ge += np.where((lnum_ro>0)&(coop_ro==1), 1, 0)
                    if isocount2to == "yes":
                        count_game_ge += 1
                        count_coop_game_ge += np.where((coop_ro==1), 1, 0)
                    count_poff_ge += poff_ro
                if ro < roound-1:
                    coop_ratio = np.divide(count_coop_game_ge, count_game_ge, where=count_game_ge>0)
                    if lorf == "leave":
                        linkmatrix = Leave_Form_tl(n=n, work=work,linkmatrix=linkmatrix,coop_ratio=coop_ratio, tl=tl)
                    elif lorf == "form":
                        linkmatrix = Leave_Form_tf(n=n, work=work,linkmatrix=linkmatrix,coop_ratio=coop_ratio, tf=tf)
                    elif lorf == "both":
                        linkmatrix = Leave_Form_tl_tf(n=n, work=work,linkmatrix=linkmatrix,coop_ratio=coop_ratio, tl=tl, tf=tf)
            print(str(tr)+"tr-"+str(ge)+"ge")
            ln = np.sum(linkmatrix,axis=1)
            cd = coop_ro
            m_random = Randomn()
            cho = Linked_choice(n=n, linkmatrix=linkmatrix)
            if lorf == "leave":
                tc,tl = Selection_tc_tl(m_random=m_random, count_poff_ge=count_poff_ge, cho=cho, tc_pre=tc, tl_pre=tl)
                tc,tl = Mutation_tc_tl(m_random=m_random,tc_pre=tc,tl_pre=tl)
            elif lorf == "form":
                tc,tf = Selection_tc_tf(m_random=m_random, count_poff_ge=count_poff_ge, cho=cho, tc_pre=tc, tf_pre=tf)
                tc,tf = Mutation_tc_tf(m_random=m_random,tc_pre=tc,tf_pre=tf)
            elif lorf == "both":
                tc,tl,tf = Selection_tc_tl_tf(m_random=m_random, count_poff_ge=count_poff_ge, cho=cho, tc_pre=tc, tl_pre=tl, tf_pre=tf)
                tc,tl,tf = Mutation_tc_tl_tf(m_random=m_random,tc_pre=tc,tl_pre=tl,tf_pre=tf)
            if lorf == "leave":
                tl_all_ges[ge] = tl
            elif lorf == "form":
                tf_all_ges[ge] = tf
            elif lorf == "both":
                tl_all_ges[ge] = tl
                tf_all_ges[ge] = tf
            tc_all_ges[ge] = tc
            ln_all_ges[ge] = ln
            cd_all_ges[ge] = cd
        if lorf == "leave":
            tl_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tl_all_ges.reshape(-1)
        elif lorf == "form":
            tf_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tf_all_ges.reshape(-1)
        elif lorf == "both":
            tl_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tl_all_ges.reshape(-1)
            tf_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tf_all_ges.reshape(-1)
        tc_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = tc_all_ges.reshape(-1)
        ln_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = ln_all_ges.reshape(-1)
        cd_all_ges_trs[tr*(n*(generation)):tr*(n*(generation))+(n*(generation))] = cd_all_ges.reshape(-1)
    tr_trs_repeat = np.repeat(np.arange(trial),(generation)*n)
    ge_ges_n = np.repeat(np.arange(generation),n)
    ge_ges_repeat = np.tile(ge_ges_n, trial)
    if lorf == "leave":
        df = pd.DataFrame({"tr":tr_trs_repeat, "ge":ge_ges_repeat, "tc":tc_all_ges_trs, "tl":tl_all_ges_trs, "ln":ln_all_ges_trs, "cd":cd_all_ges_trs})#一行にしなきゃだめみたい#, "dn":dn_all_ges_trs
    elif lorf == "form":
        df = pd.DataFrame({"tr":tr_trs_repeat, "ge":ge_ges_repeat, "tc":tc_all_ges_trs, "tf":tf_all_ges_trs, "ln":ln_all_ges_trs, "cd":cd_all_ges_trs})
    elif lorf == "both":
        df = pd.DataFrame({"tr":tr_trs_repeat, "ge":ge_ges_repeat, "tc":tc_all_ges_trs, "tl":tl_all_ges_trs, "tf":tf_all_ges_trs, "ln":ln_all_ges_trs, "cd":cd_all_ges_trs})
    df.to_csv(name+"/"+name+"_all.csv")
    time7 = time.time()
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