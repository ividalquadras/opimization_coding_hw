#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 01:09:01 2020

@author: fslataper
"""
import pandas as pd
import numpy as np
from scipy.spatial import distance
def center_calculator(df):
    centers = {}
    l = set(df["cluster"])
    for i in l:
        centers[i] = df.groupby('cluster')['point'].apply(lambda x: x.mean())[i]
    df["tuple"] = list(map(lambda x: centers[x],df["cluster"]))
    return df
from scipy.spatial import distance
def Dunn_index(df):
    if type(df["cluster"][33]) != np.ndarray:
        new_df = center_calculator(df)
        return Dunn_index(new_df)
    else:
        clusters = df.groupby(df["tuple"])
        big_delta = []
        small_delta = []
        for i in clusters.groups.keys():
            for k in df["point"][clusters.groups[i]]:
                for m in df["point"][clusters.groups[i]]:
                    big_delta.append(np.linalg.norm(np.array(k)-np.array(m)))
            s1 = list(df["point"][clusters.groups[i]])
            for j in clusters.groups.keys():
                s2 = list(df["point"][clusters.groups[j]])
                small_delta_prov = distance.cdist(s1,s2).min(axis=1)
                small_delta.append(min(small_delta_prov))
            small_delta = list(filter(lambda x: x!=0,small_delta))
        return min(small_delta)/max(big_delta)
def DB_index(df):
    if type(df["cluster"][33]) != np.ndarray:
        new_df = center_calculator(df)
        return DB_index(new_df)
    else:
        clusters = df.groupby(df["tuple"])
        dispersions = []
        separations = []
        for i in clusters.groups.keys():
            dis = []
            for m in df["point"][clusters.groups[i]]:
                dis.append(np.linalg.norm(np.array(i)-np.array(m)))
            dispersions.append((sum(dis)/len(dis))**0.5)
            seps = []
            for r in clusters.groups.keys():
                seps.append(np.linalg.norm(np.array(i)-np.array(r)))
            separations.append(seps)
        Ds = []
        for e in range(len(dispersions)):
            SoI = separations[e]
            Rs = []
            for t in range(len(SoI)):
                if SoI[t] != 0:
                    Rs.append((dispersions[e]+dispersions[t]/SoI[t]))
            Ds.append(max(Rs))
        return np.average(Ds)