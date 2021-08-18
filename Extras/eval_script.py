
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# %% [markdown]
# ### Mask-RCNN evaluate model. Balloon dataset

# %%
import os
import tqdm
import matplotlib.pyplot as plt
# os.chdir('..')
import sys
sys.path.extend([os.getcwd(),os.getcwd()+'/src'])
from samples.plates import plates
from preprocess import preprocess
from preprocess import augmentation as aug

from model import mask_rcnn_functional
import evaluating
from common import utils
from common import inference_utils
from common.inference_utils import process_input
from common.config import CONFIG
# from common.inference_optimize import maskrcnn_to_onnx, modify_onnx_model

import numpy as np
import tensorflow as tf
utils.tf_limit_gpu_memory(tf, 2000)



# %%
base_dir = r'D:\Data\cemex\patentes\maskrccnn dataset\maskrccnn dataset 500'
train_dir = base_dir
val_dir = base_dir


# %%
from common.config import CONFIG

CONFIG.update(plates.COCO_CONFIG)


# %%
eval_dataset = plates.PlateDataset(dataset_dir=base_dir,
                               subset='test',
                               # SegmentationDataset necessary parent attributes
                               augmentation=aug.get_validation_augmentation(
                                           image_size=CONFIG['img_size'],
                                           normalize=CONFIG['normalization']
                               ),
                               **CONFIG
                              )


# %%
eval_dataloader = preprocess.DataLoader(eval_dataset,
                                        shuffle=True,
                                        cast_output=False,
                                        return_original=True,
                                         **CONFIG
                                        )


# %%

# %%
# Loading inference graph and import weights
weights_path = r'D:\Source\maskrcnn_tf2\src\logs_old\scalars\maskrcnn_mobilenet_c1f61e61570ae80cd3c574c008cbf226_cp-0010.ckpt'
# weights_path = r"D:\Source\maskrcnn_tf2\src\logs_old\scalars"

inference_config = CONFIG
inference_config.update({'training': False})
inference_model = mask_rcnn_functional(config=inference_config)
inference_model = inference_utils.load_mrcnn_weights(model=inference_model,
                                                     weights_path=weights_path,
                                                     verbose=True
                                                    )


# %%
os.path.dirname(weights_path)

# %% [markdown]
# #### Evaluate data on a single batch with tensorflow

# %%
def tf_mrcnn_inference(model, infer_batch, eval_batch):
    """
    Args:
        model: tensorflow tf.keras.Model
        infer_batch: prepared data for inference
        eval_batch:  ground truth data for evaluation

    Returns: boxes,
             class_ids, 
             scores, 
             ull_masks, 
             eval_gt_boxes, 
             eval_gt_class_ids, 
             eval_gt_masks

    """

    # Extract inference inputs from dataloader
    batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,     batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = infer_batch

    # Extract original inputs from dataloader
    eval_gt_image = eval_batch[0][0]
    eval_gt_boxes = eval_batch[3][0]
    eval_gt_class_ids = eval_batch[2][0]
    eval_gt_masks = eval_batch[1][0]
    
    # Make inference
    output = model([batch_images, batch_image_meta])
    detections, mrcnn_probs, mrcnn_bbox, mrcnn_mask, rpn_rois, rpn_class, rpn_bbox = output

    # Extract bboxes, class_ids, scores and full-size masks
    boxes, class_ids, scores, full_masks =         utils.reformat_detections(detections=detections[0].numpy(),
                                  mrcnn_mask=mrcnn_mask[0].numpy(),
                                  original_image_shape=eval_gt_image.shape,
                                  image_shape=batch_images[0].shape,
                                  window=batch_image_meta[0][7:11]
                                  )
    return boxes, class_ids, scores, full_masks, eval_gt_boxes, eval_gt_class_ids, eval_gt_masks


# %%
def evaluate_mrcnn(model, inference_function, eval_dataloader, iou_limits=(0.5, 1), iou_step=0.05):
    """
    Evaluate Mask-RCNN model
    Args:
        model: tensorflow tf.keras.Model
        inference_function:
        eval_dataloader:
        iou_limits: start and end for IoU in mAP
        iou_step:   step for IoU in mAP

    Returns:

    """
    # Evaluate mAP
    for eval_iou_threshold in np.arange(iou_limits[0], iou_limits[1], iou_step):

        # Metrics lists
        ap_list = []
        precisions_list = []
        recalls_list = []

        eval_iterated = iter(eval_dataloader)
        pbar = tqdm.tqdm(eval_iterated, total=eval_dataloader.__len__())

        for eval_input, _ in pbar:
            # Split batch into prepared data for inference and original data for evaluation
            infer_batch = eval_input[:-4]
            eval_batch = eval_input[-4:]
            
            try:
                boxes, class_ids, scores, full_masks, eval_gt_boxes, eval_gt_class_ids, eval_gt_masks =                     inference_function(model=model, infer_batch=infer_batch, eval_batch=eval_batch)

                # Get AP, precisions, recalls, overlaps
                ap, precisions, recalls, overlaps =                     evaluating.compute_ap(gt_boxes=eval_gt_boxes,
                                          gt_class_ids=eval_gt_class_ids,
                                          gt_masks=eval_gt_masks,
                                          pred_boxes=boxes,
                                          pred_class_ids=class_ids,
                                          pred_scores=scores,
                                          pred_masks=full_masks,
                                          iou_threshold=eval_iou_threshold
                                          )
                postfix = ''
            except:
                postfix = 'Passed an image. AP added as zero.'
                ap = 0.0
                precisions = 0.0
                recalls = 0.0
            
            ap_list.append(ap)
            precisions_list.append(precisions)
            recalls_list.append(recalls)

            # Update tqdm mAP
            pbar.set_description(f"IoU: {eval_iou_threshold:.2f}. mAP: {np.mean(ap_list):.4f} ")# {postfix}


        print(f'mAP={np.mean(ap_list):.4f}, IoU: {eval_iou_threshold:.2f}')


# %%
evaluate_mrcnn(model=inference_model,
               inference_function=tf_mrcnn_inference,
               eval_dataloader=eval_dataloader
              )

# %% [markdown]
# #### Evaluate data on a single batch with TensorRT

# %%
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


# %%
def trt_mrcnn_inference(model, infer_batch, eval_batch):
    """

    Args:
        model: tensorflow tf.keras.Model
        infer_batch: prepared data for inference
        eval_batch:  ground truth data for evaluation

    Returns: boxes,
             class_ids, 
             scores, f
             ull_masks, 
             eval_gt_boxes, 
             eval_gt_class_ids, 
             eval_gt_masks

    """
    # Extract inference inputs from dataloader
    batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,     batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = infer_batch

    # Extract original inputs from dataloader
    eval_gt_image = eval_batch[0][0]
    eval_gt_boxes = eval_batch[3][0]
    eval_gt_class_ids = eval_batch[2][0]
    eval_gt_masks = eval_batch[1][0]

    # Extract trt-variables from a dict for transparency
    engine = model['engine']
    stream = model['stream']
    context = model['context']
    device_input = model['device_input']
    device_output1 = model['device_output1']
    device_output2 = model['device_output2']

    host_output1 = model['host_output1']
    host_output2 = model['host_output2']
    
    output_nodes = model['output_nodes']
    graph_type = model['graph_type']
    
    
    if graph_type == 'uff':
        # Prepare image for uff original graph
        input_image, window, scale, padding, crop = utils.resize_image(
                eval_gt_image,
                min_dim=800,
                min_scale=0,
                max_dim=1024,
                mode='square')
        #  Substract channel-mean
        input_image = input_image.astype(np.float32) - np.array([123.7, 116.8, 103.9])
        
        image_shape_reformat = input_image.shape
        
        # Add batch dimension
        batch_images = np.expand_dims(input_image, 0)
        # (batch, w, h, 3) -> (batch, 3, w, h)
        batch_images = np.moveaxis(batch_images, -1, 1)
        
        
        
    else:
        window = batch_image_meta[0][7:11]
        image_shape_reformat = batch_images[0].shape

    # Make inference
    host_input = batch_images.astype(dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)
    context.execute_async(bindings=[int(device_input),
                                    int(device_output1),
                                    int(device_output2),
                                    ],
                          stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(host_output1, device_output1, stream)
    cuda.memcpy_dtoh_async(host_output2, device_output2, stream)
    stream.synchronize()
    
    output_shape1 = engine.get_binding_shape(output_nodes[0])
    output_shape2 = engine.get_binding_shape(output_nodes[1])
    
    if graph_type == 'onnx':
        trt_mrcnn_detection = host_output1.reshape(output_shape1).astype(dtype=np.float32)
        trt_mrcnn_mask = host_output2.reshape(output_shape2).astype(dtype=np.float32)
    elif graph_type == 'uff':
        # (batch, 100, 6)
        trt_mrcnn_detection = host_output1.reshape(
            (engine.max_batch_size, *output_shape1)).astype(dtype=np.float32)
        # (batch, 100, 2, 28, 28)
        trt_mrcnn_mask = host_output2.reshape(
            (engine.max_batch_size, *output_shape2)).astype(dtype=np.float32)
        # (batch, 100, 2, 28, 28) -> (batch, 100, 28, 28, 2)
        trt_mrcnn_mask = np.moveaxis(trt_mrcnn_mask, 2, -1)
    else:
        raise ValueError(f'Only onnx and uff graph types. Passed: {graph_type}')
        

    # Extract bboxes, class_ids, scores and full-size masks
    trt_boxes, trt_class_ids, trt_scores, trt_full_masks =         utils.reformat_detections(detections=trt_mrcnn_detection[0],
                                  mrcnn_mask=trt_mrcnn_mask[0],
                                  original_image_shape=eval_gt_image.shape,
                                  image_shape=image_shape_reformat,
                                  window=window
                                  )
    
    return trt_boxes, trt_class_ids, trt_scores, trt_full_masks, eval_gt_boxes, eval_gt_class_ids, eval_gt_masks


# %%
def set_mrcnn_trt_engine(model_path, output_nodes=['mrcnn_detection', 'mrcnn_mask'], graph_type='onnx'):
    
    """
    Load TensorRT engine via pycuda
    Args:
        model_path: model path to TensorRT-engine
        output_nodes: output nodes names
        graph_type: onnx or uff

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
    output_shape1 = engine.get_binding_shape(output_nodes[0])
    host_output1 = cuda.pagelocked_empty(trt.volume(output_shape1) *
                                              engine.max_batch_size,
                                              dtype=np.float32)
    device_output1 = cuda.mem_alloc(host_output1.nbytes)


    # mrcnn_mask output
    output_shape2 = engine.get_binding_shape(output_nodes[1])
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
            'host_output2': host_output2,
            'output_nodes': output_nodes,
            'graph_type': graph_type
           }


# %%
evaluate_mrcnn(model=set_mrcnn_trt_engine(
    model_path=f"""../weights/maskrcnn_{CONFIG['backbone']}_512_512_3_trt_mod_fp32.engine"""),
               inference_function=trt_mrcnn_inference,
               eval_dataloader=eval_dataloader
              )


# %%
evaluate_mrcnn(model=set_mrcnn_trt_engine(
    model_path=f"""../weights/maskrcnn_{CONFIG['backbone']}_512_512_3_trt_mod_fp16.engine"""),
               inference_function=trt_mrcnn_inference,
               eval_dataloader=eval_dataloader
              )


