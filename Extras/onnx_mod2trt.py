import os
import sys

import subprocess
import cv2
import numpy as np

import matplotlib.pyplot as plt



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

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


max_batch_size = 1
# Precision mode
fp16_mode = False
# Workspace size in Mb
wspace_size = 512
trt_path = r'Extras/models/maskrcnn_mobilenet_512_512_3'
onnx_path = r'Extras/models/maskrcnn_mobilenet_512_512_3_trt_mod.onnx'


def xx(onnx_model_path, trt_model_path, fp16_mode=False):
    os.chdir('../weights')

# trtexec --onnx=bert.onnx \
#         --explicitBatch \
#         --saveEngine=bert_model.trt \
#         --minShapes=input_ids:1x128,attention_mask:1x128,token_type_ids:1x128 \
#         --optShapes=input_ids:8x128,attention_mask:8x128,token_type_ids:8x128 \
#         --maxShapes=input_ids:16x128,attention_mask:16x128,token_type_ids:16x128 \

    # Construct appropriate command
    command = [os.environ['TRTEXEC'],
            f'--onnx={onnx_model_path}',
            f'--saveEngine={trt_model_path}',
                '--workspace=2048',
                '--explicitBatch',
                '--verbose',
            ]

    # fp16 param
    if fp16_mode:
        command[2].replace('32', '16')
        command.append('--fp16')

    # tacticSources param
    # Do not neeed on jetson with aarch64 architecture for now.
    arch = os.uname().machine
    if arch == 'x86_64':
        command.append('--tacticSources=-cublasLt,+cublas')
        
    print(f'\nArch: {arch}\ntrtexec command list: {command}')

    result = subprocess.run(command, capture_output=True, check=True)
    # Print stdout inference result
    print(result.stdout.decode('utf8')[-2495:])

def onnx2trt(onnx_model_path, model_name,max_batch_size=1, fp16_mode=False, wspace_size=512):
    # Init TensorRT Logger
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    # Init TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    # Set tensorrt-prepared onnx model
    # Use explicit batch
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_builder_config() as builder_config, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        with open(onnx_model_path, 'rb') as model:
            parser.parse(model.read())

        print('Num of detected layers: ', network.num_layers)
        print('Detected inputs: ', network.num_inputs)
        print('Detected outputs: ', network.num_outputs)
        
        # Workspace size
        # 1e6 bytes == 1Mb
        builder_config.max_workspace_size = int(1e6 * wspace_size)
        
        # Precision mode
        if fp16_mode:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        
        # Max batch size
        builder.max_batch_size = max_batch_size
        
        
        # Set the list of tactic sources
        # Do not need for Jetson with aarch64 architecture for now
        arch = os.uname().machine
        if arch == 'x86_64':
            tactic_source = 1 << int(trt.TacticSource.CUBLAS) | 0 << int(trt.TacticSource.CUBLAS_LT)
            builder_config.set_tactic_sources(tactic_source)
            
        
        # Make TensorRT engine
        engine = builder.build_engine(network, builder_config)
        
        # Save TensorRT engine
        if fp16_mode:
            trt_model_name = f'{model_name}_fp16_trt.engine'
        else:
            trt_model_name = f'{model_name}_fp32_trt.engine'

        with open(trt_model_name, "wb") as f:
            f.write(engine.serialize())

# Convert!
onnx2trt(onnx_path,trt_path, max_batch_size, fp16_mode,wspace_size)

#Test
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

fp32 = 'fp16' if  fp16_mode else 'fp32'
trt_path = f"""Extras/models/maskrcnn_{CONFIG['backbone']}_512_512_3_trt_mod_{fp32}.engine"""
trt_path = "Extras/models/maskrcnn_mobilenet_512_512_3_fp16_trt.engine"

# @profile
def TestModel (trt_model, test_images_path):
    import time
    imgs = os.listdir(test_images_path)*6
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

trt_model = set_mrcnn_trt_engine(trt_path)
TestModel(trt_model, test_images_path)

# print(mem_usage)