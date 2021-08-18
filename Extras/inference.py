# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# %% [markdown]
# ### Mask-RCNN inference with tensorflow, onnxruntime, TensorRT engine.  Balloon dataset
# %% [markdown]
# 

# %%
import os
import sys
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
from common.config import CONFIG

import tensorflow as tf
utils.tf_limit_gpu_memory(tf, 1500)

# %% [markdown]
# #### Prepare model for inference

# %%
weights_path = r'models/maskrcnn_mobilenet_c1f61e61570ae80cd3c574c008cbf226_cp-0010.ckpt'


# %%
# Loading inference graph and import weights
from samples.plates import plates
# CONFIG.update(plates.COCO_CONFIG)

CONFIG.update(plates.COCO_CONFIG)

# CONFIG.update({
#                 'image_shape': (256, 256, 3),
#                 'image_resize_mode': 'square',
#                 'img_size': 256,
#                 'image_min_dim': 200,
#                 'image_min_scale': 0,
#                 'image_max_dim': 256,
#                 'batch_size': 1,
#                 'images_per_gpu': 1,

#                 }
#     )

inference_config = CONFIG
inference_config.update({'training': False})
inference_model = mask_rcnn_functional(config=inference_config)
inference_model = inference_utils.load_mrcnn_weights(model=inference_model,
                                                     weights_path=weights_path,
                                                     verbose=True
                                                    )

# %% [markdown]
# ---
# 
# #### Run several tests with tensorflow

# %%
test_images_path = r'/src/src/ttest'
os.listdir(test_images_path)


# %%
# %%time
for img_name in os.listdir(test_images_path)[:2]:
    img = cv2.imread(os.path.join(test_images_path, img_name))
    img_processed, image_meta, window = process_input(img, CONFIG)
    
    output = inference_model([np.expand_dims(img_processed, 0),
                              np.expand_dims(image_meta, 0)]
                            ) 
    
    detections, mrcnn_probs, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox = output
    
    # print(img_name, '\nOutput shapes:')
    # for out in output:
    #     print(out.shape)
    
    
    # Extract bboxes, class_ids, scores and full-size masks
    boxes, class_ids, scores, full_masks =     utils.reformat_detections(detections=detections[0].numpy(), 
                              mrcnn_mask=mrcnn_mask[0].numpy(), 
                              original_image_shape=img.shape, 
                              image_shape=img_processed.shape, 
                              window=window
                             )

    print(boxes)
    
    # fig=plt.figure(figsize=(10,10))
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
    # plt.show()    
    # # for c, s, fm in zip(class_ids, scores, np.moveaxis(full_masks, -1, 0)):

    # #     fig=plt.figure(figsize=(5,5))
    # #     plt.title(f'Mask. class_id: {c} score: {s}')
    # #     plt.imshow(fm)
    # plt.show()    

# %% [markdown]
# #### Convert model to .onnx with tf2onnx

# %%
import tf2onnx
import onnx
import onnxruntime as ort
# import onnx_graphsurgeon as gs
from common.inference_optimize import maskrcnn_to_onnx, modify_onnx_model


# %%
input_spec = (
    tf.TensorSpec((CONFIG['batch_size'], *CONFIG['image_shape']), tf.float32, name="input_image"),
    tf.TensorSpec((CONFIG['batch_size'], CONFIG['meta_shape']), tf.float32, name="input_image_meta")
)
model_name = f"""maskrcnn_{CONFIG['backbone']}_{'_'.join(list(map(str, CONFIG['image_shape'])))}""" 


# %%
maskrcnn_to_onnx(model=inference_model, 
                 model_name = model_name,
                 input_spec=input_spec,
                 kwargs={'opset': 11}
                )

# %% [markdown]
# #### Load onnx model and check it 

# %%
# Load the ONNX model
model = onnx.load(f"""../weights/maskrcnn_{CONFIG['backbone']}_512_512_3.onnx""")
# Check that the IR is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

# %% [markdown]
# #### Run several tests with onnxruntime

# %%
sess = ort.InferenceSession(f"""../weights/maskrcnn_{CONFIG['backbone']}_512_512_3.onnx""")
print(f'Inputs: {[x.name for x in sess.get_inputs()]}\nOutputs:{[x.name for x in sess.get_outputs()]}')


# %%
for img_name in os.listdir(test_images_path)[:]:
    img = cv2.imread(os.path.join(test_images_path, img_name))
    img_processed, image_meta, window = process_input(img, CONFIG)
    

    output = sess.run(output_names=[x.name for x in sess.get_outputs()], 
                      input_feed={'input_image': np.expand_dims(img_processed, 0).astype('float32'),
                                  'input_image_meta': np.expand_dims(image_meta, 0).astype('float32'),
                                 }
                     )
    
    detections, mrcnn_probs, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox = output
    
    print(img_name, '\nOutput shapes:')
    for out in output:
        print(out.shape)
    
    
    # Extract bboxes, class_ids, scores and full-size masks
    boxes, class_ids, scores, full_masks =     utils.reformat_detections(detections=detections[0], 
                              mrcnn_mask=mrcnn_mask[0], 
                              original_image_shape=img.shape, 
                              image_shape=img_processed.shape, 
                              window=window
                             )
    
    fig=plt.figure(figsize=(10,10))
    plt.title('Input data')
    plt.imshow(img)

    for c, s, fm in zip(class_ids, scores, np.moveaxis(full_masks, -1, 0)):

        fig=plt.figure(figsize=(5,5))
        plt.title(f'Mask. class_id: {c} score: {s}')
        plt.imshow(fm)

# %% [markdown]
# #### Configure model for TensorRT

# %%
modify_onnx_model(model_path=f'../weights/{model_name}.onnx',
                  config=CONFIG,
                  verbose=True
                 )

# %% [markdown]
# #### TensorRT optimization
# 
# __With trtexec:__ 

# %%
get_ipython().run_cell_magic('time', '', "\nos.chdir('../weights')\n\n# Construct appropriate command\nfp16_mode = False\ncommand = [os.environ['TRTEXEC'],\n           f'--onnx={model_name}_trt_mod.onnx',\n           f'--saveEngine={model_name}_trt_mod_fp32.engine',\n            '--workspace=2048',\n            '--explicitBatch',\n            '--verbose',\n          ]\n\n# fp16 param\nif fp16_mode:\n    command[2].replace('32', '16')\n    command.append('--fp16')\n\n# tacticSources param\n# Do not neeed on jetson with aarch64 architecture for now.\narch = os.uname().machine\nif arch == 'x86_64':\n    command.append('--tacticSources=-cublasLt,+cublas')\n    \nprint(f'\\nArch: {arch}\\ntrtexec command list: {command}')\n\nresult = subprocess.run(command, capture_output=True, check=True)\n# Print stdout inference result\nprint(result.stdout.decode('utf8')[-2495:])")

# %% [markdown]
# __With python TensorRT API:__
# 

# %%
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


# %%
max_batch_size = 1
# Precision mode
fp16_mode = False
# Workspace size in Mb
wspace_size = 2048


# %%
get_ipython().run_cell_magic('time', '', '\n# Init TensorRT Logger\nTRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)\n# Init TensorRT plugins\ntrt.init_libnvinfer_plugins(TRT_LOGGER, "")\n# Set tensorrt-prepared onnx model\nonnx_model_path = f\'../weights/{model_name}_trt_mod.onnx\'\n# Use explicit batch\nexplicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)\n\nwith trt.Builder(TRT_LOGGER) as builder, \\\n        builder.create_builder_config() as builder_config, \\\n        builder.create_network(explicit_batch) as network, \\\n        trt.OnnxParser(network, TRT_LOGGER) as parser:\n\n    with open(onnx_model_path, \'rb\') as model:\n        parser.parse(model.read())\n\n    print(\'Num of detected layers: \', network.num_layers)\n    print(\'Detected inputs: \', network.num_inputs)\n    print(\'Detected outputs: \', network.num_outputs)\n    \n    # Workspace size\n    # 1e6 bytes == 1Mb\n    builder_config.max_workspace_size = int(1e6 * wspace_size)\n    \n    # Precision mode\n    if fp16_mode:\n        builder_config.set_flag(trt.BuilderFlag.FP16)\n    \n    # Max batch size\n    builder.max_batch_size = max_batch_size\n    \n    # Set the list of tactic sources\n    # Do not need for Jetson with aarch64 architecture for now\n    arch = os.uname().machine\n    if arch == \'x86_64\':\n        tactic_source = 1 << int(trt.TacticSource.CUBLAS) | 0 << int(trt.TacticSource.CUBLAS_LT)\n        builder_config.set_tactic_sources(tactic_source)\n        \n    \n    # Make TensorRT engine\n    engine = builder.build_engine(network, builder_config)\n    \n    # Save TensorRT engine\n    if fp16_mode:\n        trt_model_name = f\'../weights/{model_name}_fp16_trt.engine\'\n    else:\n        trt_model_name = f\'../weights/{model_name}_fp32_trt.engine\'\n\n    with open(trt_model_name, "wb") as f:\n        f.write(engine.serialize())')

# %% [markdown]
# #### Run TensorRT inference

# %%
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


# %%
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


# %%
trt_model = set_mrcnn_trt_engine(f"""../weights/maskrcnn_{CONFIG['backbone']}_512_512_3_trt_mod_fp32.engine""")


# %%
for img_name in os.listdir(test_images_path):
    img = cv2.imread(os.path.join(test_images_path, img_name))
    img_processed, image_meta, window = process_input(img, CONFIG)
    

    trt_mrcnn_detection, trt_mrcnn_mask = trt_mrcnn_inference(trt_model, np.expand_dims(img_processed, 0))
    

    # Extract bboxes, class_ids, scores and full-size masks
    boxes, class_ids, scores, full_masks =     utils.reformat_detections(detections=trt_mrcnn_detection[0], 
                              mrcnn_mask=trt_mrcnn_mask[0], 
                              original_image_shape=img.shape, 
                              image_shape=img_processed.shape, 
                              window=window
                             )
    
    fig=plt.figure(figsize=(10,10))
    plt.title('Input data')
    plt.imshow(img)

    for c, s, fm in zip(class_ids, scores, np.moveaxis(full_masks, -1, 0)):

        fig=plt.figure(figsize=(5,5))
        plt.title(f'Mask. class_id: {c} score: {s}')
        plt.imshow(fm)


# %%



