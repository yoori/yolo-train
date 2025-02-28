"""
Convert VOC annotations to labelme.
"""

import sys
import copy
import typing
import argparse
import pathlib
import logging
import numpy as np
from dataclasses import dataclass
import shutil
import cv2
import lxml.etree as ET

import nx.dataset.utils


logger = logging.getLogger(__name__)


def convert_mot_annotations(  # noqa: C901
    target_root,  # < Directory for push result (jpeg + json in labelme format)
    process_root,
):
    target_root = pathlib.Path(target_root)
    voc_files = {}

    for seq_path in pathlib.Path(process_root).iterdir():
        # Process sequence
        logger.info("Process sequence: " + str(seq_path))
        images_root = seq_path / 'img1'
        ann_file = seq_path / 'det' / 'det.txt'

        frames = {}
        for image_path in images_root.rglob('*.jpg'):
            logger.info("Process image: " + str(image_path))
            try:
                frame_id = int(image_path.stem)
                frames[frame_id] = image_path
            except:
                pass

        frame_segments = {}
        with open(str(ann_file)) as f:
            for line in f:
                line = line.rstrip()
                line_arr = line.split(',')
                frame_id, obj_id, bb_left, bb_top, bb_width, bb_height = line_arr[0:6]
                bb_left = float(bb_left)
                bb_top = float(bb_top)
                bb_width = float(bb_width)
                bb_height = float(bb_height)
                frame_id = int(frame_id)
                if frame_id not in frame_segments:
                    frame_segments[frame_id] = []
                frame_segments[frame_id].append(nx.dataset.utils.Segment(
                    label="0",  # < Person
                    bbox=[bb_left, bb_top, bb_left + bb_width, bb_top + bb_height]
                ))

        target_seq_path = target_root / seq_path.name
        target_seq_path.mkdir(parents=True, exist_ok=True)

        for frame_id, image_file in frames.items():
            image = cv2.imread(image_file)
            h, w, _ = image.shape
            segments = frame_segments[frame_id] if frame_id in frame_segments else []
            normalized_segments = []
            for segment in segments:
                segment = copy.copy(segment)
                segment.bbox = [
                    segment.bbox[0] / w,
                    segment.bbox[1] / h,
                    segment.bbox[2] / w,
                    segment.bbox[3] / h
                ]
                normalized_segments.append(segment)
            shutil.copyfile(image_file, target_seq_path / (str(frame_id) + ".jpg"))
            nx.dataset.utils.write_labelme_annotations(
                target_seq_path / (str(frame_id) + ".json"),
                segments=normalized_segments,
                image_path=image_file.name,
                image_width=w,
                image_height=h,
            )


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Util for convert MOT annotations to labelme annotations.')
    parser.add_argument('--mot-root', help = 'MOT root', required = True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )
    args = parser.parse_args()
    convert_mot_annotations(
        args.target_root,
        args.mot_root,
    )
