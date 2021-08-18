import multiprocessing as mp
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
    tf_limit_gpu_memory(tf, 7500)

    # Load Mask-RCNN config
    from common.config import CONFIG

    CONFIG.update(plates.COCO_CONFIG)
    CONFIG.update({'image_shape': (256, 256, 3),
                   'image_resize_mode': 'square',
                   'img_size': 256,
                   'image_max_dim': 256,
                   'backbone': 'resnet18',
                   'epochs': 50,
                   'batch_size': 1,
                   'images_per_gpu': 1,
                   'train_bn': True,
                   'use_multiprocessing': True,
                   'workers': mp.cpu_count()

                   }
                  )

    # Set folder for coco dataset
    base_dir = r'/mnt/data/cx-ir/patentes_aug'

    # Initialize training and validation datasets
    train_dataset = plates.PlateDataset(dataset_dir=base_dir,
                                     subset='train',
                                     # SegmentationDataset necessary parent attributes
                                     augmentation=aug.get_training_augmentation(
                                         image_size=CONFIG['img_size'],
                                         normalize=CONFIG['normalization']
                                     ),
                                     **CONFIG
                                     )

    val_dataset = plates.PlateDataset(dataset_dir=base_dir,
                                   subset='valid',
                                   # SegmentationDataset necessary parent attributes
                                   augmentation=aug.get_validation_augmentation(
                                       image_size=CONFIG['img_size'],
                                       normalize=CONFIG['normalization']
                                   ),
                                   **CONFIG
                                   )

    # Init Mask-RCNN model
    CONFIG['callback']['save_dir'] = r'/home/user/MaskTests/maskrcnn_tf2/result_models/resnet18_256x256'

    model = mask_rcnn_functional(config=CONFIG)

    # Train
    train_model(model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=CONFIG,
                weights_path=None
                )
