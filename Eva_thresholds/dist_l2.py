import os

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
parser = argparse.ArgumentParser(description='face model test')
# # general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='../models/model-r34-amf/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()
#
#

def dist_l2(vec1, vec2):
    return distance.euclidean(vec1,vec2)

def evaluation(list_dists, list_match_labels,threshold):
    length = len(list_dists)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    Res = []
    for idx in range(length):
        if (list_dists[idx] <= threshold) and (list_match_labels[idx] == 1): # matching => matching
            TP += 1
        elif (list_dists[idx] > threshold ) and (list_match_labels[idx] == 1): # matching => non matching
            FN += 1
        elif (list_dists[idx] <= threshold ) and (list_match_labels[idx] == 0): #non matching => matching
            FP += 1
        else:
            TN += 1
    TPR = TP / (TP +FN ) # matching / tong_so_item_dc_xac_nhan_matching
    FPR = FP / (FP + TN)
    Acc = (TP + TN) / (TP + FN + FP + TN)
    Res.append([threshold, TPR, FPR, Acc])
    Res = np.array(Res)
    d = {'Threshold': Res[:,0], 'TPR': Res[:,1], 'FPR': Res[:,2], 'Acc': Res[:,3]}
    return pd.DataFrame(data=d)

with open("list_image_path.pkl","rb") as op:
    data = pickle.load(op)
# print(len(data))
model = face_embedding.FaceModel(args)
list_dist = []
list_match_label = []

for key,value in data.items():

    list_image_nonmatch = value[0]
    list_image_match = value[1]
    image_main_path = value[2]

    image_main = cv2.imread(image_main_path[0])
    # print(image_main)
    vec_image_main = model.get_feature(image_main)
    print('main1',type(vec_image_main))
    if (vec_image_main is None):
        continue
    ### Non-match
    for image_path_non in list_image_nonmatch:
        img = cv2.imread(image_path_non)
        feature1 = model.get_feature(img)
        print('feat1',type(feature1))
        if (feature1 is None):
            continue
        dist = dist_l2(vec_image_main,feature1)
        list_dist.append(dist)
        list_match_label.append(0)
    ### Match
    for image_path_match in list_image_match:
        img_ = cv2.imread(image_path_match)
        # print(img_)
        feature2 = model.get_feature(img_)
        print('feat2',type(feature2))
        if (feature2 is None):
            continue
        dist_ = dist_l2(vec_image_main,feature2)
        list_dist.append(dist_)
        list_match_label.append(1)

with open("distl2.pkl","wb") as dis:
    pickle.dump(list_dist,dis)
with open("label_matching.pkl","wb") as label:
    pickle.dump(list_match_label,label)

# for threshold in np.arange(0.7, 1.5, 0.02):
#     test = evaluation(list_dist,list_match_label,threshold)
#     print(test)


# model = face_embedding.FaceModel(args)
# time_now = time.perf_counter()
# img = cv2.imread('Tom_Hanks_54745.png')
# f1 = model.get_feature(img)
# time_now2 = time.perf_counter()
# diff = time_now2 - time_now
#
# print(f1)
# print(f1.shape)
# print(diff)

# data_path = '../recognition/datasets/faces_emore_image/'
# # for id in os.listdir(data_path):
# #     path_id = os.path.join(data_path,id)
# #
# #     if len(os.listdir(path_id)) < 5:
# #         print(len(os.listdir(path_id)))
# #         print(path_id)
# data = Prepare(data_path,3,3)
# test = data.image_match()

print("Done!!!!!!!")
