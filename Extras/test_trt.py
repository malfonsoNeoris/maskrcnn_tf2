import os
import sys

import cv2
import numpy as np

# import tensorflow as tf
# utils.tf_limit_gpu_memory(tf, 500)

sys.path.extend([os.getcwd(),os.getcwd()+'/src'])

from common import inference_utils
from common.inference_utils import process_input
from common import utils


from common.config import CONFIG
from samples.plates import plates
CONFIG.update(plates.COCO_CONFIG)

test_images_path = r'src/ttest'
os.listdir(test_images_path)

from memory_profiler import profile
from memory_profiler import memory_usage

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


def trt_mrcnn_inference(model, image):
    """

    Args:
        model: tensorflow tf.keras.Model
        image: prepared image for inference

    Returns: boxes,
             class_ids, 
             scores, f
             ull_masks, 
             eval_gt_boxes, 
             eval_gt_class_ids, 
             eval_gt_masks

    """

    # Extract trt-variables from a dict for transparency
    engine = model['engine']
    stream = model['stream']
    context = model['context']
    device_input = model['device_input']
    device_output1 = model['device_output1']
    device_output2 = model['device_output2']

    host_output1 = model['host_output1']
    host_output2 = model['host_output2']

    # Make inference
    host_input = image.astype(dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async(bindings=[int(device_input),
                                    int(device_output1),
                                    int(device_output2),
                                    ],
                          stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(host_output1, device_output1, stream)
    cuda.memcpy_dtoh_async(host_output2, device_output2, stream)
    stream.synchronize()
    
    trt_mrcnn_detection = host_output1.reshape(
        engine.get_binding_shape('mrcnn_detection')).astype(dtype=np.float32)
    trt_mrcnn_mask = host_output2.reshape(
        engine.get_binding_shape('mrcnn_mask')).astype(dtype=np.float32)
    
    return trt_mrcnn_detection, trt_mrcnn_mask

# @profile
def set_mrcnn_trt_engine(model_path):
    
    """
    Load TensorRT engine via pycuda
    Args:
        model_path: model path to TensorRT-engine

    Returns: python dict of attributes for pycuda model inference

    """
    
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(trt_logger, "")

    with open(model_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Inputs
    input_shape = engine.get_binding_shape('input_image')
    input_size = trt.volume(input_shape) *\
                 engine.max_batch_size * np.dtype(np.float32).itemsize
    device_input = cuda.mem_alloc(input_size)

    # Outputs
    output_names = list(engine)[1:]

    # mrcnn_detection output
    output_shape1 = engine.get_binding_shape('mrcnn_detection')
    host_output1 = cuda.pagelocked_empty(trt.volume(output_shape1) *
                                              engine.max_batch_size,
                                              dtype=np.float32)
    device_output1 = cuda.mem_alloc(host_output1.nbytes)


    # mrcnn_mask output
    output_shape2 = engine.get_binding_shape('mrcnn_mask')
    host_output2 = cuda.pagelocked_empty(trt.volume(output_shape2) * engine.max_batch_size,
                                              dtype=np.float32)
    device_output2 = cuda.mem_alloc(host_output2.nbytes)

    # Setting a cuda stream
    stream = cuda.Stream()
    
    return {'engine': engine,
            'stream': stream,
            'context': context,
            'device_input': device_input,
            'device_output1': device_output1,
            'device_output2':device_output2,
            'host_output1': host_output1,
            'host_output2': host_output2
           }

trt_path = "Extras/models/maskrcnn_mobilenet_512_512_3_fp16_trt_work 1024.engine"
trt_path = 'Extras/models/maskrcnn_mobilenet_512_512_3_fp32_trt.engine'
trt_model = set_mrcnn_trt_engine(trt_path)

# @profile
def TestModel (trt_model, test_images_path):
    import time
    imgs = os.listdir(test_images_path)*10
    t = time.perf_counter()

    for img_name in imgs :
        img = cv2.imread(os.path.join(test_images_path, img_name))
        img_processed, image_meta, window = process_input(img, CONFIG)
        

        trt_mrcnn_detection, trt_mrcnn_mask = trt_mrcnn_inference(trt_model, np.expand_dims(img_processed, 0))
        

        # Extract bboxes, class_ids, scores and full-size masks
        boxes, class_ids, scores, full_masks = \
        utils.reformat_detections(detections=trt_mrcnn_detection[0], 
                                mrcnn_mask=trt_mrcnn_mask[0], 
                                original_image_shape=img.shape, 
                                image_shape=img_processed.shape, 
                                window=window
                                )
        
        # fig=plt.figure(figsize=(10,10))
        # plt.title('Input data')
        # plt.imshow(img)

        # for c, s, fm in zip(class_ids, scores, np.moveaxis(full_masks, -1, 0)):

        #     fig=plt.figure(figsize=(5,5))
        #     plt.title(f'Mask. class_id: {c} score: {s}')
        #     plt.imshow(fm)
    
    dt = time.perf_counter()-t

    print('time spent: ', dt, ' for ', len(imgs), ' images. Mean FPS=', len(imgs)/dt)

TestModel(trt_model, test_images_path)

# print(mem_usage)