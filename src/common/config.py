import os
import multiprocessing as mp
import numpy as np

CLASS_DICT = {'background': 0, 'balloon': 1}
CLASSES_NUM = len(CLASS_DICT.keys())

CONFIG = {

    # For now use NHWC - channel last
    # meta_shape =
    #              image_id +
    #              3 channel original image shape +
    #              3 channel image shape+
    #              4 coordinates of window+,
    #              scale number+,
    #              number of classes

    'image_shape': (512, 512, 3),
    'img_size': 512,
    'backbone': 'mobilenet',
    'meta_shape': (1 + 3 + 3 + 4 + 1 + CLASSES_NUM),
    'num_classes': CLASSES_NUM,
    'class_dict': CLASS_DICT,

    # Image normalization
    # ImageNet: {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    'normalization': None,

    'image_min_dim': 300,
    'image_min_scale': 0,
    'image_max_dim': 512,
    'image_resize_mode': 'square',

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    'use_mini_masks': False,
    'mini_mask_shape': (32, 32),

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    'mask_shape': (28, 28),

    'epochs': 100,
    'gpu_num': 1,
    'batch_size': 1,
    'images_per_gpu': 1,
    'training': True,
    'log_per_steps': 5,
    'use_multiprocessing': True,
    'workers': mp.cpu_count()//2,

    'callback': {
        # TensorBoard callback
        'log_dir': os.path.join('logs', 'scalars'),
        'save_dir': os.path.join('logs', 'scalars'),
        # ReduceLROnPlateau callback
        'reduce_lr_on_plateau': 0.98,
        'reduce_lr_on_plateau_patience': 10,
        # ModelCheckpoint callback
        'save_weights_only': True,
        'save_best_only': True,
        'histogram_freq': 0,
        'profile_batch': '1,2',
    },

    'backbone_strides': [4, 8, 16, 32, 64],

    'top_down_pyramid_size': 256,

    # Length of square anchor side in pixels
    'rpn_anchor_scales': (32, 64, 128, 256, 512),

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    'rpn_anchor_ratios': [0.5, 1, 2],

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    'rpn_anchor_stride': 1,

    'rpn_train_anchors_per_image': 256,
    'max_gt_instances': 100,

    # Bounding box refinement standard deviation for RPN and final detections.
    'rpn_bbox_std_dev': np.array([0.1, 0.1, 0.2, 0.2], dtype='float32'),
    'bbox_std_dev': np.array([0.1, 0.1, 0.2, 0.2], dtype='float32'),

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more proposals.
    'rpn_nms_threshold': 0.7,

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    'use_rpn_rois': True,

    'random_rois': 0,

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    'detection_min_confidence': 0.7,
    # Non-maximum suppression threshold for detection
    'detection_nms_threshold': 0.3,
    # Max number of final detections
    'detection_max_instances': 100,

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    'pre_nms_limit': 6000,  # 1024,  # 6000

    # ROIs kept after non-maximum suppression (training and inference)
    'post_nms_rois_training': 2000,
    'post_nms_rois_inference': 1000,

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    'train_rois_per_image': 200,

    # Percent of positive ROIs used to train classifier/mask heads
    'roi_positive_ratio': 0.33,

    # Pooled ROIs
    'pool_size': 7,
    'mask_pool_size': 14,

    # Size of the fully-connected layers in the classification graph
    'fpn_cls_fc_layers_size': 1024,

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    # Order: rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss
    'loss_weights': [1, 1, 1, 1, 1],

    # Optimizer config
    'optimizer_kwargs': {
        'learning_rate': 0.001,
        # 'clipnorm': 5.0,
        'clipvalue': 5.0,
        'name': 'adamax',
    },

    # L2 regularization param
    'weight_decay': 0.0002,
    'train_bn': False,
    'l2_reg_batchnorm': False,

}
