"""
Download dataset from https://cocodataset.org/#download :
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
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

import nx.dataset.utils


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
	level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(description='Util for convert COCO annotations to labelme annotations.')
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
            str(i + 1): str(i) for i in range(9)
        },
    )
