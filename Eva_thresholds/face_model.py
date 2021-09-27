from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import insightface
from insightface.utils import face_align
from mtcnn_detector import MtcnnDetector
import face_preprocess
def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, prefix, epoch, layer):
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, ctx_id, model_prefix, model_epoch, det_type=0, use_large_detector=False):
        self.det_type = det_type
        self.det_threshold = [0.6, 0.7, 0.8]
        if use_large_detector:
            self.detector = insightface.model_zoo.get_model('retinaface_r50_v1')
        else:
            if det_type == 0:
                detector = MtcnnDetector(model_folder='mtcnn-model',
                                         ctx=mx.gpu(ctx_id),
                                         num_worker=1,
                                         accurate_landmark=True,
                                         threshold=self.det_threshold)
            else:
                detector = MtcnnDetector(model_folder=mtcnn_path,
                                         ctx=mx.gpu(ctx_id),
                                         num_worker=1,
                                         accurate_landmark=True,
                                         threshold=[0.0, 0.0, 0.2])
            self.detector = detector
        #self.detector.prepare(ctx_id=ctx_id)
        if ctx_id>=0:
            ctx = mx.gpu(ctx_id)
        else:
            ctx = mx.cpu()
        image_size = (112,112)
        self.model = get_model(ctx, image_size, model_prefix, model_epoch, 'fc1')
        self.image_size = image_size

    def get_input(self, face_img):
        ret = self.detector.detect_face(face_img, det_type=self.det_type)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        bbox = bbox[0, 0:4]
        points = points[0, :].reshape((2, 5)).T
        #print(bbox)
        #print(points)
        nimg = face_preprocess.preprocess(face_img,
                                          bbox,
                                          points,
                                          image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2, 0, 1))
        return aligned

    def get_feature(self, aligned):
        a = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        a = np.transpose(a, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()[0]
        norm = np.sqrt(np.sum(emb*emb)+0.00001)
        emb /= norm
        return emb

