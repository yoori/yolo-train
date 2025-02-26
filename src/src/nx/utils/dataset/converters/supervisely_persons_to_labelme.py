"""
Convert "Supervisely persons" annotations to labelme.

For download set you can use:
import dataset_tools as dtools
dtools.download(dataset='Supervisely Persons', dst_dir='~/dataset-ninja/')
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

    parser = argparse.ArgumentParser(
        description='Util for convert "Supervisely Persons" annotations to labelme annotations.'
    )
    parser.add_argument('--root', help = 'Supervisely Persons root', required = True)
    parser.add_argument('--model', help='model', type=str, required=True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )
    args = parser.parse_args()

    model = ultralytics.YOLO(args.model)
    for sub_dir in pathlib.Path(args.root).glob('*'):
        if sub_dir.is_dir():
            target_sub_dir = pathlib.Path(args.target_root) / sub_dir.name
            target_sub_dir.mkdir(parents=True, exist_ok=True)
            nx.dataset.utils.datasetninja_convert(
                target_sub_dir,
                sub_dir / "img",  # < Images root.
                sub_dir / "ann",  # < Annotations root.
                model=model,
                class_mapping={
                    'person_bmp': 0,
                    'person_poly': 0,
                },
                ignore_files_with_classes=[
                    'bicycle', '1',
                    'car', '2',
                    'motorcycle', '3',
                    'aircraft', '4',
                    'train', '6',
                    'truck', '7',
                    'boat', '8',
                ],
            )
