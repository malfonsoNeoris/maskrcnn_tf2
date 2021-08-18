
import os
import sys

import subprocess
import cv2
import numpy as np

import matplotlib.pyplot as plt
import time
import tf2onnx
import onnx
import onnxruntime as ort

sys.path.extend([os.getcwd(),os.getcwd()+'/src'])

from common import inference_utils
from common.inference_utils import process_input
from common import utils

import tensorflow as tf
utils.tf_limit_gpu_memory(tf, 500)

from common.config import CONFIG
from samples.plates import plates
CONFIG.update(plates.COCO_CONFIG)


test_images_path = r'src/ttest'
os.listdir(test_images_path)

model_path = 'Extras/models/maskrcnn_mobilenet_512_512_3.onnx'

# # Load the ONNX model
# model = onnx.load(model_path)
# # Check that the IR is well formed
# onnx.checker.check_model(model)
# # Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

# %%
sess = ort.InferenceSession(model_path)
print(f'Inputs: {[x.name for x in sess.get_inputs()]}\nOutputs:{[x.name for x in sess.get_outputs()]}')

imgs = os.listdir(test_images_path)*10
t = time.perf_counter()
for img_name in imgs:
    img = cv2.imread(os.path.join(test_images_path, img_name))
    img_processed, image_meta, window = process_input(img, CONFIG)
    

    output = sess.run(output_names=[x.name for x in sess.get_outputs()], 
                      input_feed={'input_image': np.expand_dims(img_processed, 0).astype('float32'),
                                  'input_image_meta': np.expand_dims(image_meta, 0).astype('float32'),
                                 }
                     )
    
    detections, mrcnn_probs, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox = output
    
    # print(img_name, '\nOutput shapes:')
    # for out in output:
    #     print(out.shape)
    
    
    # Extract bboxes, class_ids, scores and full-size masks
    boxes, class_ids, scores, full_masks =     utils.reformat_detections(detections=detections[0], 
                              mrcnn_mask=mrcnn_mask[0], 
                              original_image_shape=img.shape, 
                              image_shape=img_processed.shape, 
                              window=window
                             )
    
    # plt.title('Input data')

    # plt.imshow(img, 'gray', interpolation='none')
    # out_data = zip(class_ids, scores, np.moveaxis(full_masks, -1, 0))
    # out_data = sorted(filter(lambda x: x[1]>=.9,out_data), key=lambda y: y[1], reverse=True)
    # # data= next(out_data)
    # # if data is None or not any(data):
    # #     continue
    # c, s, fm = out_data[0]

    # plt.imshow(fm, 'jet', interpolation='none', alpha=0.3)
    # plt.title(f'Mask. class_id: {c} score: {s}')
    # plt.savefig('pepe.jpg')   

dt = time.perf_counter()-t
print('time spent: ', dt, ' for ', len(imgs), ' images. Mean FPS=', len(imgs)/dt)


