import os
import matplotlib.pyplot as plt
import face_embedding
import argparse
import cv2
import numpy as np
import pandas as pd
import datetime
import time
from prepare_data import Prepare
import pickle
from scipy.spatial import distance

def evaluation(list_dists, list_match_labels,thresholds):
    Res = []
    for threshold in thresholds:
        length = len(list_dists)
        TP = 0
        FN = 0
        FP = 0
        TN = 0

        for idx in range(length):
            if (list_dists[idx] <= threshold) and (list_match_labels[idx] == 1): # matching => matching
                TP += 1
            elif (list_dists[idx] > threshold ) and (list_match_labels[idx] == 1): # matching => non matching
                FN += 1
            elif (list_dists[idx] <= threshold ) and (list_match_labels[idx] == 0): #non matching => matching
                FP += 1
            else:
                TN += 1
        # for idx in range(length):
        #     if (list_dists[idx] >= threshold) and (list_match_labels[idx] == 1): # matching => matching
        #         TP += 1
        #     elif (list_dists[idx] < threshold ) and (list_match_labels[idx] == 1): # matching => non matching
        #         FN += 1
        #     elif (list_dists[idx] >= threshold ) and (list_match_labels[idx] == 0): #non matching => matching
        #         FP += 1
        #     else:
        #         TN += 1
        TPR = TP / (TP +FN ) # matching / tong_so_item_dc_xac_nhan_matching
        FPR = FP / (FP + TN)
        Acc = (TP + TN) / (TP + FN + FP + TN)
        Res.append([threshold, TPR, FPR, Acc])

    Res = np.array(Res)
    d = {'Threshold': list(Res[:,0]), 'TPR': list(Res[:,1]), 'FPR': list(Res[:,2]), 'Acc': list(Res[:,3])}
    return pd.DataFrame(data=d)

with open("distl2.pkl","rb") as dis:
     list_dist = pickle.load(dis)
with open("label_matching.pkl","rb") as label:
     list_match_label = pickle.load(label)
#
thresholds = np.arange(0.7, 1.5, 0.02)
test = evaluation(list_dist,list_match_label,thresholds)
print(test)
print(type(test))
#ax = plt.gca()
#test.plot(kind='line',y='TPR',x='Threshold',color='blue',ax=ax)
#test.plot(kind='line',y='FPR',x='Threshold',color='red',ax=ax)
#test.plot(kind='line',y='Acc',x='Threshold',color='yellow',ax=ax)
#plt.savefig('output.png')

# for threshold in np.arange(0.7, 1.5, 0.02):
#      test = evaluation(list_dist,list_match_label,threshold)
#      print(test)

#with open("dist_cosine.pkl","rb") as dis:
#     list_dist = pickle.load(dis)
#with open("label_matching_cosine.pkl","rb") as label:
#     list_match_label = pickle.load(label)
#
#for threshold in np.arange(0.0, 1.0, 0.01):
 #    test = evaluation(list_dist,list_match_label,threshold)
  #   print(test)
