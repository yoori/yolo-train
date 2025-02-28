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
        description='Util for convert Objects365 dataset annotations to labelme annotations.'
    )
    parser.add_argument('--images-root', help = 'Images root', required = True)
    parser.add_argument('--ann-root', help = 'Annotations root', required = True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )

    args = parser.parse_args()

    nx.dataset.utils.coco_convert(
	pathlib.Path(args.target_root),
        pathlib.Path(args.ann_root),  # < Annotations root.
        pathlib.Path(args.images_root),  # < Images root.
        class_mapping={
            '0': 0,
            'Person': 0,

            '46': 1,
            'Bicycle': 1,

            '5': 2,
            'Car': 2,
            '49': 2,
            'Van': 2,
            '87': 2,
            'Pickup Truck': 2,
            '126': 2,
            'Sports Car': 2,

            '58': 3,
            'Motorcycle': 3,

            '114': 4,
            'Airplane': 4,

            '55': 5,
            'Bus': 5,

            '116': 6,
            'Train': 6,

            '65': 7,
            'Truck': 7,
            '188': 7,
            'Fire Truck': 7,
            '199': 7,
            'Heavy Truck': 7,

            '21': 8,
            'Boat': 8,
        },
    )
