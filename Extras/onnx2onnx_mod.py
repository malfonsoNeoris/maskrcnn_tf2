import os
import sys

import subprocess
import cv2
import numpy as np

import matplotlib.pyplot as plt

import tf2onnx
import onnx
import onnxruntime as ort

sys.path.extend([os.getcwd(),os.getcwd()+'/src'])
# import onnx_graphsurgeon as gs
from common.inference_optimize import maskrcnn_to_onnx, modify_onnx_model

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
modify_onnx_model(model_path=model_path,
                  config=CONFIG,
                  verbose=True
                 )
