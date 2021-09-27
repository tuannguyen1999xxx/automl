from prepare_data import Prepare
import pickle
import cv2
import numpy as np

data_path = 'datasets/faces_emore_image'
# for id in os.listdir(data_path):
#     path_id = os.path.join(data_path,id)
#
#     if len(os.listdir(path_id)) < 5:
#         print(len(os.listdir(path_id)))
#         print(path_id)
data = Prepare(data_path,5,5)
test = data.image_match()

print("Done!!!!!!!")

