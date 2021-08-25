# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %% [markdown]
# ### Mask-RCNN inference with tensorflow, onnxruntime, TensorRT engine.  Balloon dataset
# %% [markdown]
# 

# %%
import os
from re import T
import sys
import random
import onnx
sys.path.extend([os.getcwd(),os.getcwd()+'/src'])


# %%
import subprocess
import cv2
import numpy as np

import matplotlib.pyplot as plt

from layers import losses
from training import get_optimizer
from model import mask_rcnn_functional
from common import inference_utils
from common.inference_utils import process_input
from common import utils

import tensorflow as tf
# utils.tf_limit_gpu_memory(tf, 1500)

# %% [markdown]
# #### Prepare model for inference

# %%

from samples.plates import plates

# Tensorflow model
def GetConfig(update_from, backbone = 'resnet50', img_size = 256, image_min_dim = 200):
    from common.config import CONFIG
    CONFIG.update(update_from)

    CONFIG.update({'image_shape': (img_size, img_size, 3),
                   'image_resize_mode': 'square',
                   'img_size': img_size,
                   'image_min_dim': img_size,
                   'image_max_dim': img_size,
                   'backbone': backbone,
                   'batch_size': 1,
                   'images_per_gpu': 1,
                   }
                  )
    return CONFIG
def LoadTensorlowModel(config, weights_path):
    config.update({'training': False})
    inference_model = mask_rcnn_functional(config=config)
    inference_model = inference_utils.load_mrcnn_weights(model=inference_model,
                                                        weights_path=weights_path,
                                                        verbose=True
                                                        )
    return inference_model

def model_infer(img, config, model, model_type='tf'):

    img_processed, image_meta, window = process_input(img, config)

    output=None
    if model_type == 'tf':
        output= model([np.expand_dims(img_processed, 0),
                                np.expand_dims(image_meta, 0)]
                                )
    elif model_type=='onnx':
        outs = [x.name for x in model.get_outputs()]
        output = model.run(output_names=outs, 
                        input_feed={'input_image': np.expand_dims(img_processed, 0).astype('float32'),
                                    'input_image_meta': np.expand_dims(image_meta, 0).astype('float32'),
                                    }
                        )    
    elif model_type=='trt':
        trt_mrcnn_detection, trt_mrcnn_mask = InferWithEngine(np.expand_dims(img_processed, 0), model)
        output = (trt_mrcnn_detection, None, None, trt_mrcnn_mask, None, None, None)
    else:
        raise Exception('model invalid')

    detections, _, _, mrcnn_mask, _, _, _ = output
            
    # Extract bboxes, class_ids, scores and full-size masks
    boxes, class_ids, scores, full_masks =     utils.reformat_detections(detections=detections[0].numpy(), 
                            mrcnn_mask=mrcnn_mask[0].numpy(), 
                            original_image_shape=img.shape, 
                            image_shape=img_processed.shape, 
                            window=window
                            )

    return boxes, class_ids, scores, full_masks
def TestTensorflowModel (tests_path, inference_config, inference_model):
    
    import time
    t = time.perf_counter()
    imgs = os.listdir(tests_path)[:20]

    results = []

    for img_n, img_name in enumerate(imgs):
        img = cv2.imread(os.path.join(tests_path, img_name))
        img_processed, image_meta, window = process_input(img, inference_config)

        output = model_infer(img, inference_config, inference_model)
    

        out_data = zip( class_ids, scores, boxes, np.moveaxis(full_masks, -1, 0))
        out_data = sorted(filter(lambda x: x[1]>=.9,out_data), key=lambda y: y[1], reverse=True)
        N = len(out_data)
        colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(N)]
        img_overlay = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        for index, (c, s, b, fm) in enumerate (out_data ):
            color = colors[index]
            mask_image = cv2.merge([ fm*color[0],fm*color[1],fm*color[2]]).astype(np.uint8)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2BGRA)
            img_overlay = cv2.addWeighted(img_overlay, .4, mask_image, .6,0)
            y1,x1,y2,x2 = b
            img_overlay= cv2.rectangle(img_overlay, (x1,y1), (x2,y2), color, 2)


        cv2.imwrite(f'test_{img_n}.jpg', img_overlay)
    

    dt = time.perf_counter()-t
    print ('dt: ', dt, ' -  Images: ', len(imgs), ' -  FPS: ', len(imgs)/dt)

# Onnx Model
def Convert2OnnxModel(config,inference_model, base_folder):
    from common.inference_optimize import maskrcnn_to_onnx

    input_spec = (
        tf.TensorSpec((config['batch_size'], *config['image_shape']), tf.float32, name="input_image"),
        tf.TensorSpec((config['batch_size'], config['meta_shape']), tf.float32, name="input_image_meta")
    )
    output_path = os.path.join(base_folder,f"""maskrcnn_{config['backbone']}_{'_'.join(list(map(str, config['image_shape'])))}.onnx""" )

    maskrcnn_to_onnx(model=inference_model, 
                    output_path = output_path,
                    input_spec=input_spec,
                    kwargs={'opset': 11}
                    )
def LoadAndCheckOnnxModel(config, model_path):
    import onnx

    # #### Load onnx model and check it 

    # Load the ONNX model
    model = onnx.load(model_path)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))
def LoadOnnxModel( weight_path):
    import onnxruntime as ort

    sess = ort.InferenceSession(weight_path)
    print(f'Inputs: {[x.name for x in sess.get_inputs()]}\nOutputs:{[x.name for x in sess.get_outputs()]}')
    return sess
def InferWithOnnx(img, onnx_session,config):
    outs = [x.name for x in onnx_session.get_outputs()]

    img_processed, image_meta, window = process_input(img, config)        
    output = onnx_session.run(output_names=outs, 
                        input_feed={'input_image': np.expand_dims(img_processed, 0).astype('float32'),
                                    'input_image_meta': np.expand_dims(image_meta, 0).astype('float32'),
                                    }
                        )
    return output

# TRT model
def Convert2TrtModel(config, onnx_path, max_batch_size=1,fp16_mode=False, wspace_size=512 ):
    from common.inference_optimize import modify_onnx_model
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda

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

            base_dir = os.path.dirname(onnx_model_path)
            
            with open(os.path.join(base_dir, trt_model_name), "wb") as f:
                f.write(engine.serialize())
            print('succefully created .engine')

    modify_onnx_model(model_path=onnx_path,
                    config=config,
                    verbose=True
                    )
    
    onnx_mod_path = onnx_path.replace('.onnx', '_trt_mod.onnx')
    if os.path.exists(onnx_mod_path):
        model_name = f"""maskrcnn_{config['backbone']}_{'_'.join(list(map(str, config['image_shape'])))}"""
        onnx2trt(onnx_mod_path, model_name, max_batch_size=max_batch_size, fp16_mode=fp16_mode, wspace_size=wspace_size)   
def LoadTrtModel(model_path):
    
    """
    Load TensorRT engine via pycuda
    Args:
        model_path: model path to TensorRT-engine

    Returns: python dict of attributes for pycuda model inference

    """
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(trt_logger, "")

    with open(model_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Inputs
    input_shape = engine.get_binding_shape('input_image')
    input_size = trt.volume(input_shape) *                 engine.max_batch_size * np.dtype(np.float32).itemsize
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
def InferWithEngine(image, model):
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
    import pycuda.autoinit
    import pycuda.driver as cuda
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
#----------------------------------------
def model_infer(img, config, model, model_type='tf'):

    img_processed, image_meta, window = process_input(img, config)

    output=None
    if model_type == 'tf':
        output= model([np.expand_dims(img_processed, 0),
                                np.expand_dims(image_meta, 0)]
                                )
    elif model_type=='onnx':
        outs = [x.name for x in model.get_outputs()]
        output = model.run(output_names=outs, 
                        input_feed={'input_image': np.expand_dims(img_processed, 0).astype('float32'),
                                    'input_image_meta': np.expand_dims(image_meta, 0).astype('float32'),
                                    }
                        )    
    elif model_type=='trt':
        trt_mrcnn_detection, trt_mrcnn_mask = InferWithEngine(np.expand_dims(img_processed, 0), model)
        output = (trt_mrcnn_detection, None, None, trt_mrcnn_mask, None, None, None)
    else:
        raise Exception('model invalid')

    detections, _, _, mrcnn_mask, _, _, _ = output
            
    # Extract bboxes, class_ids, scores and full-size masks
    boxes, class_ids, scores, full_masks =     utils.reformat_detections(detections=detections[0].numpy(), 
                            mrcnn_mask=mrcnn_mask[0].numpy(), 
                            original_image_shape=img.shape, 
                            image_shape=img_processed.shape, 
                            window=window
                            )

    return boxes, class_ids, scores, full_masks
def TestModel(tests_path, inference_config, inference_model, model_type='tf'):
    
    import time
    t = time.perf_counter()
    imgs = os.listdir(tests_path)[:20]

    for img_n, img_name in enumerate(imgs):
        img = cv2.imread(os.path.join(tests_path, img_name))

        boxes, class_ids, scores, full_masks = model_infer(img, inference_config, inference_model, model_type=model_type)
    

        out_data = zip( class_ids, scores, boxes, np.moveaxis(full_masks, -1, 0))
        out_data = sorted(filter(lambda x: x[1]>=.9,out_data), key=lambda y: y[1], reverse=True)
        N = len(out_data)
        colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(N)]
        img_overlay = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        for index, (c, s, b, fm) in enumerate (out_data ):
            color = colors[index]
            mask_image = cv2.merge([ fm*color[0],fm*color[1],fm*color[2]]).astype(np.uint8)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2BGRA)
            img_overlay = cv2.addWeighted(img_overlay, .4, mask_image, .6,0)
            y1,x1,y2,x2 = b
            img_overlay= cv2.rectangle(img_overlay, (x1,y1), (x2,y2), color, 2)


        cv2.imwrite(f'test_{img_n}.jpg', img_overlay)
    

    dt = time.perf_counter()-t
    print ('dt: ', dt, ' -  Images: ', len(imgs), ' -  FPS: ', len(imgs)/dt)

#---------------------

weigths_paths=[
    (r'result_models/mobilenet_256x256_500/maskrcnn_mobilenet_732675f3f3e77a4c2a80db439ab9dc9e_cp-0010.ckpt','mobilenet', 256),
    (r'result_models/mobilenet_256x256_50003/maskrcnn_mobilenet_987dbebfa7028966383175f23dae5b89_cp-0011.ckpt','mobilenet', 256),
    (r'result_models/mobilenet_512x512_50003/maskrcnn_mobilenet_c1f61e61570ae80cd3c574c008cbf226_cp-0010.ckpt','mobilenet', 512),
    (r'result_models/resnet18_256x256_500/maskrcnn_resnet18_16a5e7ed4b511704027fb29c476f9928_cp-0012.ckpt','resnet18', 256),
    (r'result_models/resnet18_256x256_aug/maskrcnn_resnet18_026ec169f9153c7945ce40682ab82e1d_cp-0001.ckpt','resnet18', 256),
    (r'result_models/resnet18_512x512_500/maskrcnn_resnet18_fd8ee0523a839b3db1b9562253c43a41_cp-0019.ckpt','resnet18', 512),
    ]


#%% Set what to DO
TENSORFLOW_TEST=True
TENSOR2ONNX_CONVERT=False
ONNX_TEST = False
ONNX2TRT=False
TRT_TEST=False
selected = 2
test_images_path = r'/data/cx-ir/patentes_500/images/'



weigths_path, backbone, img_size = weigths_paths[selected]
inference_config = GetConfig(plates.COCO_CONFIG, backbone=backbone,img_size=img_size)
if TENSORFLOW_TEST or TENSOR2ONNX_CONVERT:
    inference_model = LoadTensorlowModel(inference_config, weigths_path)
if TENSORFLOW_TEST:
    TestModel(test_images_path, inference_config, inference_model, model_type='tf')

#%% Convert The model!
if TENSOR2ONNX_CONVERT:
    base_folder = os.path.dirname(weigths_path)
    Convert2OnnxModel(inference_config,inference_model,base_folder )

#%% Testing ONNX model!
if ONNX_TEST or ONNX2TRT or TRT_TEST:
    base_folder = os.path.dirname(weigths_path)
    onnx_path = os.path.join(base_folder,f"""maskrcnn_{inference_config['backbone']}_{'_'.join(list(map(str, inference_config['image_shape'])))}.onnx""" )
    # onnx_path = os.path.join(base_folder,f"""maskrcnn_{inference_config['backbone']}_{'_'.join(list(map(str, inference_config['image_shape'])))}_trt_mod.onnx""" )


if ONNX_TEST :
    onnx_sess = LoadOnnxModel( onnx_path)
    TestModel(test_images_path,inference_config,  onnx_sess, model_type='onnx')

if ONNX2TRT:
    Convert2TrtModel(inference_config, onnx_path)

if TRT_TEST:
    model_name = f"""maskrcnn_{inference_config['backbone']}_{'_'.join(list(map(str, inference_config['image_shape'])))}"""
    ttype = 'fp32'
    trt_model_name = f'{model_name}_{ttype}_trt.engine'

    trt_path = os.path.join(base_folder, trt_model_name)
    trt_model  =LoadTrtModel(trt_path)
    TestModel(test_images_path,inference_config,  trt_model, model_type='trt')


