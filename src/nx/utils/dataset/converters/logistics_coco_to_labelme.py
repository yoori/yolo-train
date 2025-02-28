"""
Download dataset from
"""

import os
import sys
import copy
import json
import pathlib
import collections
import logging
import numpy as np
import shutil
import argparse

import ultralytics

import nx.dataset.utils


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
	level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(
        description='Util for convert Logistics dataset annotations to labelme annotations.'
    )
    parser.add_argument('--images-root', help = 'Images root', required = True)
    parser.add_argument('--ann-root', help = 'Annotations root', required = True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )
    parser.add_argument('--model', help='model', type=str, required=True)

    args = parser.parse_args()

    model = ultralytics.YOLO(args.model)

    nx.dataset.utils.coco_convert(
	pathlib.Path(args.target_root),
        pathlib.Path(args.ann_root),  # < Annotations root.
        pathlib.Path(args.images_root),  # < Images root.
        class_mapping={
            'person': 0,
            'car': 2,
            'van': 2,
            'truck': 7,
        },
        model=model,
        ignore_files_with_classes=[
            'bicycle', '1',
            'motorcycle', '3',
            'airplane', '4',
            'bus', '5',
            'train', '6',
            'boat', '8',
        ],
    )
