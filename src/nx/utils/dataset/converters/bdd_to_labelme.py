"""
Convert BDD100K segmentation annotations to labelme.

About set: it contains much scenes with much cars.

For download BDD100K you can use:
import dataset_tools as dtools
dtools.download(dataset='BDD100K: Images 100K', dst_dir='~/bdd/')

BDD root folder contains train, test, val folders - need apply this script to 'train' and 'val' folders separetly.

BDD contains much bbox annotations (and little polygons) - convert bboxes to labelme rectangle.
trainer stop geometric rect unsafe transformations for files that contains bbox.
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

    parser = argparse.ArgumentParser(description='Util for convert BDD annotations to labelme annotations.')
    parser.add_argument('--bdd-root', help = 'BDD root', required = True)
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
        pathlib.Path(args.bdd_root) / "img",  # < Images root.
        pathlib.Path(args.bdd_root) / "ann",  # < Annotations root.
        model=model,
        class_mapping={
            'person': 0,
            'rider': 0,
            'bike': 1,
            'car': 2,
            'motor': 3,
            'bus': 5,
            'train': 6,
            'truck': 7,
        },
        ignore_files_with_classes=['aeroplane', '4', 'boat', '8'],
    )
