import os
import random

import tensorflow as tf

from common.utils import tf_limit_gpu_memory
from model import mask_rcnn_functional
from preprocess import augmentation as aug
from samples.plates import plates
from training import train_model

if __name__ == '__main__':
    # Init random seed
    random.seed(42)

    # Limit GPU memory for tensorflow container
    tf_limit_gpu_memory(tf, 4500)

    # Load Mask-RCNN config
    from common.config import CONFIG

    CONFIG.update(plates.COCO_CONFIG)

    CONFIG.update({
                   'image_shape': (256, 256, 3),
                   'image_resize_mode': 'square',
                   'img_size': 256,
                   'image_min_dim': 128,
                   'image_min_scale': 0,
                   'image_max_dim': 256,

                #    'backbone': 'mobilenet',
                   'epochs': 10,
                #    'batch_size': 1,
                #    'images_per_gpu': 1,
                #    'train_bn': False,

                   }
        )

    # Init training and validation datasets
    base_dir = r'D:\Data\cemex\patentes\maskrccnn dataset\maskrccnn dataset 500'
    train_dir = base_dir
    val_dir = base_dir

    # Initialize training and validation datasets

    train_dataset = plates.PlateDataset(dataset_dir=train_dir,
                                     subset='train',
                                     # SegmentationDataset necessary parent attributes
                                     augmentation=aug.get_training_augmentation(
                                         image_size=CONFIG['img_size'],
                                         normalize=CONFIG['normalization']
                                     ),
                                     **CONFIG
                                     )

    val_dataset = plates.PlateDataset(dataset_dir=val_dir,
                                   subset='valid',
                                   # SegmentationDataset necessary parent attributes
                                   augmentation=aug.get_validation_augmentation(
                                       image_size=CONFIG['img_size'],
                                       normalize=CONFIG['normalization']
                                   ),
                                   **CONFIG
                                   )

    # Use only 1000 random images for train and 100 random images for validation
    # train_imgs = 1000
    # val_imgs = 100
    # random.shuffle(train_dataset.images_names)
    # random.shuffle(val_dataset.images_names)
    # train_dataset.images_names = train_dataset.images_names[:train_imgs]
    # val_dataset.images_names = val_dataset.images_names[:val_imgs]

    # Init Mask-RCNN model
    model = mask_rcnn_functional(config=CONFIG)
    CONFIG['callback']['save_dir'] = r'D:\Source\maskrcnn_tf2\src\result_models\256x256'
    # Train
    train_model(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=CONFIG,
                weights_path=None)
