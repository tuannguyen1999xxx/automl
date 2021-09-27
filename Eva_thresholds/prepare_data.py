import os
import cv2
import numpy as np
import pickle
import random
# data_path = '../recognition/datasets/faces_emore_image/'
# list_ids = os.listdir(data_path)
# print(len(list_ids))
#
# def image_match():
#     list_ids = os.listdir(data_path)
#     print(list_ids)

class Prepare:
    def __init__(self, data_path, num_match, num_nonmatch):
        self.num_match = num_match
        self.num_nonmatch = num_nonmatch
        self.data_path = data_path

    def list_path_nonmatch(self,ids):
        list_path = []
        random.shuffle(ids)
        ids_choose = ids[:self.num_nonmatch]

        for id in ids_choose:
            id_path = os.path.join(self.data_path,id)
            for count, image_name in enumerate(os.listdir(id_path)):
                if count > 0:
                    continue
                image_path = os.path.join(id_path,image_name)
                list_path.append(image_path)

        return list_path

    def image_match(self):

        dict_data = {}
        ids_check = os.listdir(self.data_path)
        for count, ids in enumerate(os.listdir(self.data_path)):
            if count > 1000:
                break
            list_ID_data = []
            list_main = []
            temp_path = os.path.join(self.data_path,ids)
            if (len(os.listdir(temp_path)) < self.num_match*2):
                continue
            image_names = os.listdir(temp_path)
            random.shuffle(image_names)
            ### Add data match
            # if (len(image_names) < self.num_match):
            #     list_match = image_names
            # print(image_names)
            list_match = [os.path.join(temp_path, x) for x in image_names[:self.num_match]]
            main_image = os.path.join(temp_path,image_names[self.num_match + 1])
            list_main.append(main_image)
            ### Add data non-match
            if ids in ids_check:
                ids_check.remove(ids)
            list_nonmatch = self.list_path_nonmatch(ids_check)

            list_ID_data.append(list_nonmatch)
            list_ID_data.append(list_match)
            list_ID_data.append(list_main)
            ### Data[0] -> nonmatch data[1] -> match data[2]: one vector to match
            print(count)
            # for image_name in os.listdir(temp_path):
            #     image_path = os.path.join(temp_path,image_name)
            dict_data[ids] = list_ID_data
        with open("list_image_path.pkl","wb") as w:
            pickle.dump(dict_data,w)
        return dict_data