"""
Convert "Vehicle for YOLO" annotations to labelme.

For download set you can use:
import dataset_tools as dtools
dtools.download(dataset='Vehicle Dataset for YOLO', dst_dir='~/dataset-ninja/')
"""

import sys
import argparse
import pathlib
import logging

import ultralytics

import nx.dataset.utils


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(description='Util for convert "vehicle for yolo" annotations to labelme annotations.')
    parser.add_argument('--root', help = 'Vehicle for yolo root', required = True)
    parser.add_argument('--model', help='model', type=str, required=True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )
    args = parser.parse_args()

    model = ultralytics.YOLO(args.model)
    nx.dataset.utils.datasetninja_convert(
        args.target_root,
        pathlib.Path(args.root) / "img",  # < Images root.
        pathlib.Path(args.root) / "ann",  # < Annotations root.
        model=model,
        class_mapping={
            'bus': 5,
            'car': 2,
            'van': 2,
            'motorbike': 3,
            'threewheel': 3,
            'truck': 7,
        },
        ignore_files_with_classes=[
            'person', '0',
            'bicycle', '1',
            'airplane', '4',
            'train', '6',
            'boat', '8'
        ],
    )
