"""
Convert VOC annotations to labelme.
"""

import sys
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


VOC_CLASS_COLORS = {
    (192, 128, 128): 0,  # person
    (0, 128, 0): 1,  # bicycle
    (128, 128, 128): 2,  # car
    (64, 128, 128): 3,  # motorbike
    (128, 0, 0): 4,  # aeroplane
    (0, 128, 128): 5,  # bus
    (128, 192, 0): 6,  # train
    # truck ?
    (0, 0, 128): 8,  # boat
}


VOC_CLASS_NAMES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorbike': 3,
    'aeroplane': 4,
    'bus': 5,
    'train': 6,
    'boat': 8,
}


JOIN_MASK_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5),
)


@dataclass
class VOCFile(object):
    image_file: pathlib.Path
    mask_file: pathlib.Path = None
    object_mask_file: pathlib.Path = None
    annotations_file: pathlib.Path = None


def load_voc_mask(file_path, object_mask_file):
    result_segments = []

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    object_mask_image = cv2.imread(object_mask_file)
    object_mask_image = cv2.cvtColor(object_mask_image, cv2.COLOR_BGR2RGB)

    for color, class_index in VOC_CLASS_COLORS.items():
        c1 = np.array(list(color), dtype=np.uint8)  # np.array(list(color), dtype=np.uint8)
        c2 = np.array(list(color), dtype=np.uint8)  # np.array(list(color), dtype=np.uint8)
        mask = cv2.inRange(image, c1, c2)
        if mask.sum() > 0:
            # Class found on image - parse objects of this class.
            objects_mask_image = cv2.bitwise_and(object_mask_image, object_mask_image, mask=mask)
            objects_colors = np.unique(objects_mask_image.reshape(-1, objects_mask_image.shape[2]), axis=0)

            for object_color in objects_colors:
                if not (object_color == np.array([0, 0, 0], dtype=object_color.dtype)).all():
                    object_mask = cv2.inRange(objects_mask_image, object_color, object_color)
                    contours, _ = cv2.findContours(object_mask, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_SIMPLE)
                    object_segments = []
                    for contour in contours:
                        contour = contour.squeeze()
                        if len(contour) > 2:
                            object_segments.append([[int(p[0]), int(p[1])] for p in contour])
                    if len(object_segments) > 1:
                        result_segments.append(nx.dataset.utils.Segment(
                            label=str(class_index),
                            polygon=nx.dataset.utils.merge_multi_segment(object_segments)
                        ))
                    elif len(object_segments) > 0:
                        result_segments.append(nx.dataset.utils.Segment(
                            label=str(class_index),
                            polygon=object_segments[0]
                        ))
                    else:
                        logger.debug("Skipped segment on " + str(file_path))
    return result_segments


def load_voc_annotations(file_path):
    try:
        result_segments = []
        txt = pathlib.Path(file_path).read_text()
        el_tree = ET.fromstring(txt)
        for obj in el_tree.findall('.//object'):
            class_name = obj.find('.//name').text.strip().lower()
            if class_name in VOC_CLASS_NAMES:
                bbox_el = obj.find('./bndbox')  # < Use only object level bbox, bndbox can be present inside 'part' elements.
                xmin_el = int(float(bbox_el.find('.//xmin').text.strip()))
                ymin_el = int(float(bbox_el.find('.//ymin').text.strip()))
                xmax_el = int(float(bbox_el.find('.//xmax').text.strip()))
                ymax_el = int(float(bbox_el.find('.//ymax').text.strip()))
                result_segments.append(nx.dataset.utils.Segment(
                    label=str(VOC_CLASS_NAMES[class_name]),
                    bbox=[xmin_el, ymin_el, xmax_el, ymax_el],
                ))
        return result_segments
    except Exception as e:
        raise Exception("Error on reading '" + str(file_path) + "' annotations file") from e


def convert_voc_annotations(  # noqa: C901
    target_root,  # < Directory for push result (jpeg + json in labelme format)
    images_root,
    annotations_root,
    mask_root,  # < Directory with *.png segments annotations
    object_mask_root,
):
    voc_files = {}

    for path in pathlib.Path(images_root).rglob('*.jpg'):
        voc_files[path.stem] = VOCFile(image_file=path)
        logger.debug("Found image file: " + str(path))

    for path in pathlib.Path(mask_root).rglob('*.png'):
        if path.stem in voc_files:
            voc_files[path.stem].mask_file = path
            logger.debug("Found mask file: " + str(path))

    for path in pathlib.Path(object_mask_root).rglob('*.png'):
        if path.stem in voc_files:
            voc_files[path.stem].object_mask_file = path
            logger.debug("Found object mask file: " + str(path))

    for path in pathlib.Path(annotations_root).rglob('*.xml'):
        if path.stem in voc_files:
            voc_files[path.stem].annotations_file = path
            logger.debug("Found annotations file: " + str(path))

    # Skip files without mask or annotations.
    for key, voc_file in voc_files.items():
        segments: typing.List[nx.dataset.utils.Segment] = None

        if voc_file.mask_file is not None and voc_file.object_mask_file is not None:
            segments = load_voc_mask(voc_file.mask_file, voc_file.object_mask_file)
        elif voc_file.annotations_file is not None:
            segments = load_voc_annotations(voc_file.annotations_file)

        image_file = voc_file.image_file
        image = cv2.imread(image_file)
        h, w, _ = image.shape

        # Normalize segments coordinates.
        for segment in segments:
            segment.polygon = [[p[0] / w, p[1] / h] for p in segment.polygon]

        # Filter invisible segments (low size for example).
        filtered_segments = []
        for s in segments:
            if nx.dataset.utils.filter_segment(s, width=w, height=h):
                filtered_segments.append(s)
            else:
                logger.debug("Drop invisible segment on " + str(voc_file.image_file))
        segments = filtered_segments

        # Check that image contains any required label.
        if segments is None or len(segments) == 0:
            print("Ignore")
            continue  # < Ignore voc file.

        shutil.copyfile(image_file, pathlib.Path(target_root) / (image_file.name))

        # Save lebelme markup to file.
        result_labelme_file = str(pathlib.Path(target_root) / pathlib.Path(image_file.stem + ".json"))
        logger.debug(
            "Save annotations for '" + str(image_file.name) + "' to '" +
            result_labelme_file + "'"
        )
        nx.dataset.utils.write_labelme_annotations(
            result_labelme_file,
            segments=segments,
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

    parser = argparse.ArgumentParser(description='Util for convert VOC annotations to labelme annotations.')
    parser.add_argument('--voc-root', help = 'VOC root', required = True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )
    args = parser.parse_args()
    convert_voc_annotations(
        args.target_root,
        pathlib.Path(args.voc_root) / "JPEGImages",  # < Images root.
        pathlib.Path(args.voc_root) / "Annotations",  # < Annotations root.
        pathlib.Path(args.voc_root) / "SegmentationClass",  # < Class mask files root.
        pathlib.Path(args.voc_root) / "SegmentationObject",  # < Object mask files root.
    )
