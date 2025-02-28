"""
"""

import os
import sys
import copy
import random
import collections
import contextlib
import tempfile
import jinja2
import logging
import multiprocessing
import argparse
import pathlib
import time
import datetime

import torch
import torch.quantization
import ultralytics
import ultralytics.utils
import ultralytics.data.build
import ultralytics.data.utils
import ultralytics.nn.autobackend

import nx.dataset.base_dataset
import nx.train.utils
from nx.dataset.albumentation_dataset import AlbumentationDataset
from nx.dataset.dali_kornia_dataset import DaliKorniaDataset


logger = logging.getLogger(__name__)


NX_YOLO_MODEL_CONFIG = """
# Parameters
nc: 9 # number of classes

scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs

#  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
#  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
#  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
#  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]  # level #20
  - [[-1, 10], 1, Concat, [1]] # cat head P5  # level #21
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)  # level #22

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)  # level #23
"""

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def load_model(model_file, weights_file, data_file):
    logger.info("To load model")
    with suppress_stdout():
        data = ultralytics.data.utils.check_det_dataset(data_file)
        val_model = ultralytics.nn.tasks.DetectionModel(model_file)
        val_model.load(ultralytics.nn.tasks.attempt_load_one_weight(weights_file)[0])
        device = torch.device('cuda:0')
        val_model = ultralytics.nn.autobackend.AutoBackend(
            weights=val_model,
            device=device,
            dnn=False,
            data=data,
            fp16=False,
        )
        val_model.eval()
        imgsz = 640
        val_model.warmup(imgsz=(1, 3, imgsz, imgsz))
    return val_model
    

def load_dataset(dataset_root):
    logger.info("To load dataset")
    dataset = DaliKorniaDataset(
        data={
            'names': {
                0: 'person',
                1: 'bicycle',
                2: 'car',
                3: 'motorcycle',
                4: 'airplane',
                5: 'bus',
                6: 'train',
                7: 'truck',
                8: 'boat',
            }
        },
        img_path=dataset_root,
        imgsz=640,
        augment=False,
    )
    logger.info("Dataset loaded")
    return dataset


def prepare_batch(validator, items):
    device = torch.device('cuda:0')
    batch = nx.dataset.base_dataset.collate_fn(items)
    # Run only actual steps from ultralytics.engine.validator.BaseValidator.__call__
    batch = validator.preprocess(batch)
    for k in ["img", "cls", "bboxes", "batch_idx"]:
        batch[k] = batch[k].to(device)
    return batch

    
def eval_mAP(validator, val_model, preds, batch):
    validator.init_metrics(val_model)
    preds = validator.postprocess(preds)
    validator.update_metrics(preds, batch)
    validator.get_stats()  # < Required for update metrics
    validator.finalize_metrics()
    local_mAP = validator.metrics.results_dict['metrics/mAP50-95(B)']
    return local_mAP


def min_val(model_file, weights_file, data_file, dataset_root, percent=0.1):
    device = torch.device('cuda:0')
    val_model = load_model(model_file, weights_file, data_file)
    dataset = load_dataset(dataset_root)
    validator = nx.train.utils.create_validator(dataset_root)

    mAP_to_label = nx.train.utils.get_labels_mAP()
    """
    mAP_to_label = {}
    labels = dataset.get_labels()
    for label_index in range(len(labels)):
        # Run only actual steps from ultralytics.engine.validator.BaseValidator.__call__
        if label_index % 1000 == 0:
            logger.info("Processed " + str(label_index) + "/" + str(len(labels)) + " images")
        items = dataset.__getitems__([label_index])
        batch = prepare_batch(validator, items)
        preds = val_model(batch["img"], augment=False)
        detect_preds = preds[0]  # < Use only detection part.

        # Filter detections for only expected classes (for use util with standard yolo models).
        # preds : 0-3, 4: confidence, 5: class
        filtered_preds = []
        for p in detect_preds.squeeze().permute(1, 0).tolist():
            class_index = round(p[5])
            if class_index < 9:
                filtered_preds.append(p)
            else:
                filtered_preds.append([0] * len(p))
        detect_preds = torch.FloatTensor([filtered_preds])
        detect_preds = detect_preds.permute(0, 2, 1)
        detect_preds = detect_preds.to(device)

        local_mAP = eval_mAP(validator, val_model, [detect_preds], batch)
        mAP_to_label[(local_mAP, label_index)] = items[0]
    """

    result = []
    cur_index = 0
    for k, label in sorted(mAP_to_label.items()):
        if cur_index > len(labels) * percent:
            break
        result.append((label, k[0]))
        cur_index += 1

    return result


def prepare_args_parser():
    parser = argparse.ArgumentParser(description='Default yolo11 model training script.')
    subparsers = parser.add_subparsers(dest='command', help='subcommand help')

    # Configure 'validate' command.
    validate_parser = subparsers.add_parser('validate-min')
    validate_parser.add_argument('--model', help='load model', type=str, required=True)
    validate_parser.add_argument(
        '-d', '--dataset-root',
        help = 'data set root',
        required=True
    )
    validate_parser.add_argument(
        '--mode', help='mode', type=str,
        choices=[
            'albumentation',
            'dali',
            'yolo'
        ],
        default='dali'
    )

    return parser


def main():  # noqa: C901
    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    ultralytics.utils.LOGGER.setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)

    parser = prepare_args_parser()
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix='.yaml') as modelFile, \
        tempfile.NamedTemporaryFile(suffix='.yaml') as data_file:  # noqa: E125
        logger.info("Model file: " + str(modelFile.name))
        with open(modelFile.name, mode='w') as f:
            f.write(NX_YOLO_MODEL_CONFIG)

        with open(data_file.name, mode='w') as f:
            data_conf_str = jinja2.Template("""
path: {{datasetRoot}} # dataset root dir
train: train # train images (relative to 'path') 4 images
val: . # val images (relative to 'path') 4 images
test: # test images (optional)
scale: n

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
"""
            ).render({"datasetRoot": args.dataset_root})  # noqa: E124
            f.write(data_conf_str)

        if args.command == 'validate-min':
            result_labels = min_val(modelFile.name, args.model, data_file.name, args.dataset_root)
            for result_label, mAP in result_labels:
                image_file = result_label['im_file']
                print(image_file + " " + str(mAP))


if __name__ == "__main__":
    main()
