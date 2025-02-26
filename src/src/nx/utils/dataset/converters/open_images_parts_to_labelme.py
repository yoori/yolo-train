"""
Convert 'Open Images v6' annotations to labelme annotated set.

Open Images is very big, but dirty dataset:
  * here much unlabeled entities of required types (person, car, ...)
  * here much grouped entities, and group non marked with IsGroup
  * incorrectly annotated objects
We use only manually labeled entities (cut it from image).

For download 'Open Images v6' set you can use:
import fiftyone
dataset = fiftyone.zoo.load_zoo_dataset("open-images-v6", split="train",
  label_types=["detections", "segmentations"],
  classes=["Person","Bicycle","Car","Ambulance","Golf cart","Taxi","Snowmobile","Motorcycle","Airplane","Bus",
    "Train","Truck","Boat"])
dataset = fiftyone.zoo.load_zoo_dataset("open-images-v6", split="validation",
  label_types=["detections", "segmentations"],
  classes=["Person","Bicycle","Car","Ambulance","Golf cart","Taxi","Snowmobile","Motorcycle","Airplane","Bus",
    "Train","Truck","Boat"])
dataset = fiftyone.zoo.load_zoo_dataset("open-images-v6", split="test",
  label_types=["detections", "segmentations"],
  classes=["Person","Bicycle","Car","Ambulance","Golf cart","Taxi","Snowmobile","Motorcycle","Airplane","Bus",
    "Train","Truck","Boat"])
"""

import typing
import sys
import argparse
import pathlib
import logging
import json
import csv
import dataclasses
import uuid
import math
import numpy as np

import cv2
import ultralytics

import nx.dataset.utils


logger = logging.getLogger(__name__)
TARGET_W = 640
TARGET_H = 640


@dataclasses.dataclass
class OISegment(object):
    label: str
    mask_file: pathlib.Path = None
    bbox: typing.Tuple[float, float, float, float] = None
    segment: nx.dataset.utils.Segment = None
    additional_options: typing.Dict = dataclasses.field(default_factory=list)
    confidence: float = 0
    is_group: bool = False


class OIAnnotatedFile(object):
    image_file: pathlib.Path
    bboxes: typing.List[OISegment]
    segments: typing.List[OISegment]

    def __init__(self, image_file):
        self.image_file = image_file
        self.bboxes = []
        self.segments = []


def iou(box1, box2) -> float:
    x_a = max(box1[0], box2[0])
    y_a = max(box1[1], box2[1])
    x_b = min(box1[2], box2[2])
    y_b = min(box1[3], box2[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    boxa_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxb_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return (
        inter_area / float(boxa_area + boxb_area - inter_area)
        if boxa_area + boxb_area - inter_area > 0 else 0
    )


def process_hierarchy_childs(hierarchy_json, class_mapping, current_parent_label):
    result_class_mapping = {}
    if "LabelName" in hierarchy_json:
        current_label = hierarchy_json["LabelName"]
        if current_label in class_mapping:
            current_parent_label = current_label
        if current_parent_label is not None:
            result_class_mapping[current_label] = class_mapping[current_parent_label]
        if "Subcategory" in hierarchy_json:
            for child in hierarchy_json["Subcategory"]:
                result_class_mapping.update(
                    process_hierarchy_childs(child, class_mapping, current_parent_label)
                )
    return result_class_mapping


def enrich_class_mapping_with_hierarchy(hierarchy_file, class_mapping):
    file_content = pathlib.Path(hierarchy_file).read_text()
    hierarchy_json = json.loads(file_content)
    result_class_mapping = dict(class_mapping)
    result_class_mapping.update(
        process_hierarchy_childs(hierarchy_json, class_mapping, None)
    )
    return result_class_mapping


def open_images_convert(
    target_root,
    root,
    class_mapping = {},
    allow_grouped_objects: bool = False,
    allow_non_verified: bool = False,
    model = None,
    skip_confidence_threshold = 0.23,
    skip_max_iou = 0.8,
):
    STEP_H = 40
    STEP_W = 40
    ignore_result_images_with_classes = {}
    for i in range(9):
        ignore_result_images_with_classes[str(i)] = True

    mask_files = {}  # < Mask file name to mask file path mapping.
    for path in pathlib.Path(root / 'labels' / 'masks').rglob('*.png'):
        mask_files[path.name] = path

    files = {}

    for path in pathlib.Path(root / 'data').rglob('*.jpg'):
        files[path.stem] = OIAnnotatedFile(image_file=path)

    # Load metadata/hierarchy.json for map child classes to required parent class.
    class_mapping = enrich_class_mapping_with_hierarchy(
        root / 'metadata' / 'hierarchy.json', class_mapping
    )

    # Load class id by class name.
    class_id_to_name = {}

    with open(root / 'metadata' / 'classes.csv', mode ='r') as file:
        csv_file = csv.reader(file)
        for row in csv_file:
            class_id_to_name[row[0]] = row[1]

    # Load bbox csv by labels/detections.csv.
    logger.info("To load detections.csv")
    loaded_bbox_count = 0
    with open(root / 'labels' / 'detections.csv', mode ='r') as file:
        csv_file = csv.DictReader(file)
        for line in csv_file:
            class_id = line['LabelName']
            if class_id not in class_id_to_name:
                continue
            image_id = line['ImageID']
            class_name = class_id_to_name[class_id]
            if image_id in files and class_name in class_mapping:
                confidence = float(line['Confidence'])
                is_group_of = int(line['IsGroupOf'])
                min_x = max(float(line['XMin']), 0)
                min_y = max(float(line['YMin']), 0)
                max_x = min(float(line['XMax']), 1)
                max_y = min(float(line['YMax']), 1)
                is_truncated = line['IsTruncated']
                additional_options = {}
                if str(is_truncated) == '1':
                    additional_options['is_truncated'] = True
                elif str(is_truncated) == '0':
                    additional_options['is_truncated'] = False
                files[image_id].bboxes.append(OISegment(
                    label=str(class_mapping[class_name]),
                    bbox=(min_x, min_y, max_x, max_y),
                    additional_options=additional_options,
                    confidence=confidence,
                    is_group=(is_group_of > 0),
                ))
                loaded_bbox_count += 1
    logger.info("Loaded " + str(loaded_bbox_count) + " bboxes (after class filtration)")

    # Load segments
    logger.info("To load segmentations.csv")
    loaded_segements_count = 0
    with open(root / 'labels' / 'segmentations.csv', mode ='r') as file:
        csv_file = csv.DictReader(file)
        for line in csv_file:
            class_id = line['LabelName']
            if class_id not in class_id_to_name:
                continue
            image_id = line['ImageID']
            class_name = class_id_to_name[class_id]
            if image_id in files and class_name in class_mapping:
                min_x = max(float(line['BoxXMin']), 0)
                min_y = max(float(line['BoxYMin']), 0)
                max_x = min(float(line['BoxXMax']), 1)
                max_y = min(float(line['BoxYMax']), 1)
                mask_file = line['MaskPath']
                if mask_file in mask_files:
                    files[image_id].segments.append(OISegment(
                        label=str(class_mapping[class_name]),
                        mask_file=mask_files[mask_file],
                        bbox=(min_x, min_y, max_x, max_y),
                    ))
                    loaded_segements_count += 1
                else:
                    logger.error("Can't find mask file by mask name: " + mask_file)
                    del files[image_id]
    logger.info("Loaded " + str(loaded_segements_count) + " segments")

    # Process files.
    logger.info("To process files")
    bboxes_dropped_by_segments = 0
    bboxes_dropped_by_criteria = 0
    bboxes_dropped_as_blocked = 0
    bboxes_dropped_by_prediction = 0
    number_of_files = len(files.keys())
    for file_i, (image_id, oi_file) in enumerate(files.items()):
        if file_i > 0 and file_i % 1000 == 0:
            logger.info("Processed " + str(file_i) + "/" + str(number_of_files) + " files")
        logger.debug("Process file: " + str(image_id))

        # Remove bboxes that have segment annotation,
        # get segment confidence by confidence defined in bbox.
        filtered_bboxes = []
        for bbox in oi_file.bboxes:
            max_iou_i = None
            max_iou = 0
            for segment_i, segment in enumerate(oi_file.segments):
                if bbox.label == segment.label:
                    assert bbox.bbox is not None
                    result_iou = iou(bbox.bbox, segment.bbox)
                    if result_iou > max_iou:
                        max_iou = result_iou
                        max_iou_i = segment_i
            if max_iou_i is not None and max_iou > 0.95:
                segment_for_bbox = oi_file.segments[max_iou_i]
                segment_for_bbox.additional_options = bbox.additional_options
                segment_for_bbox.confidence = bbox.confidence
                segment_for_bbox.is_group = bbox.is_group
                logger.debug("Drop bbox covered by segment")
                bboxes_dropped_by_segments += 1
            else:
                filtered_bboxes.append(bbox)
        oi_file.bboxes = filtered_bboxes

        logger.debug(
            "File " + str(image_id) + " contains " + str(len(oi_file.bboxes)) + " bboxes and " +
            str(len(oi_file.segments)) + " segments"
        )

        # get polygon by mask for segment.
        result_segments = []
        for oi_segment in oi_file.segments:
            object_mask_image = cv2.imread(oi_segment.mask_file)
            h, w, _ = object_mask_image.shape
            object_mask = cv2.inRange(object_mask_image, (127, 127, 127), (255, 255, 255))
            contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            object_segments = []
            for contour in contours:
                contour = contour.squeeze()
                if len(contour) > 2:
                    object_segments.append(
                        [[min(max(p[0] / w, 0), 1), min(max(p[1] / h, 0), 1)] for p in contour]
                    )

            if len(object_segments) > 1:
                oi_segment.segment = nx.dataset.utils.Segment(
                    label=str(oi_segment.label),
                    polygon=nx.dataset.utils.merge_multi_segment(object_segments),
                    additional_options=oi_segment.additional_options,
                )
                result_segments.append(oi_segment)
            elif len(object_segments) > 0:
                oi_segment.segment = nx.dataset.utils.Segment(
                    label=str(oi_segment.label),
                    polygon=object_segments[0],
                    additional_options=oi_segment.additional_options,
                )
                result_segments.append(oi_segment)

        oi_file.segments = result_segments

        for bbox_segment in oi_file.bboxes:
            bbox_segment.segment = nx.dataset.utils.Segment(
                label=str(bbox_segment.label),
                bbox=bbox_segment.bbox,
                additional_options=bbox_segment.additional_options,
            )

        oi_file.segments += oi_file.bboxes
        oi_file.bboxes = []

        # Cut bboxes and segments, for that:
        #   confidence >= 0.999 (human verified)
        #   is_group == False
        #   no cross with other labeled objects and predicted objects.
        image = cv2.imread(oi_file.image_file)
        h, w, _ = image.shape

        non_blocked_segments = []
        for check_oi_segment in oi_file.segments:
            if (
                not check_oi_segment.is_group and
                check_oi_segment.confidence >= 0.999
            ):
                block_candidates = [
                    oi_segment.segment for oi_segment in
                    filter(lambda s: s != check_oi_segment, oi_file.segments)
                ]
                try:
                    if nx.dataset.utils.blocked_area_percentage(check_oi_segment.segment, block_candidates) < 0.01:
                        non_blocked_segments.append(check_oi_segment)
                    else:
                        bboxes_dropped_as_blocked += 1
                except Exception:
                    logger.exception("Error on processing: " + str(oi_file.image_file))
            else:
                bboxes_dropped_by_criteria += 1
        if len(non_blocked_segments) == 0:
            logger.debug("Ignore file without non filtered segments: " + str(oi_file.image_file.name))
            continue

        for result_oi_segment in non_blocked_segments:
            object_image = None
            if result_oi_segment.segment.polygon is not None:
                # for segmented object drop background
                polygon_in_pixels = [[int(p[0] * w), int(p[1] * h)] for p in result_oi_segment.segment.polygon]
                min_x = min(polygon_in_pixels, key=lambda p: p[0])[0]
                max_x = max(polygon_in_pixels, key=lambda p: p[0])[0]
                min_y = min(polygon_in_pixels, key=lambda p: p[1])[1]
                max_y = max(polygon_in_pixels, key=lambda p: p[1])[1]
                result_image_height = max_y - min_y
                result_image_width = max_x - min_x
                if result_image_height > 0 and result_image_width > 0:
                    mask_image = np.zeros((result_image_height, result_image_width, 3), dtype=np.uint8)
                    object_image = image[min_y:max_y, min_x:max_x]
                    fill_poly = [[p[0] - min_x, p[1] - min_y] for p in polygon_in_pixels]
                    cv2.fillPoly(
                        mask_image,
                        pts=[np.array(fill_poly)],
                        color=(255, 255, 255)
                    )
                    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
                    ret, mask_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
                    object_image = cv2.bitwise_and(object_image, object_image, mask=mask_image)
                    result_segment = nx.dataset.utils.Segment(
                        label=result_oi_segment.label,
                        polygon=[
                            [(p[0] * w - min_x) / result_image_width, (p[1] * h - min_y) / result_image_height]
                            for p in result_oi_segment.segment.polygon
                        ],
                        additional_options=result_oi_segment.additional_options,
                    )
            else:
                bbox = result_oi_segment.segment.bbox
                min_x = int(bbox[0] * w)
                max_x = int(bbox[2] * w)
                min_y = int(bbox[1] * h)
                max_y = int(bbox[3] * h)
                result_image_height = max_y - min_y
                result_image_width = max_x - min_x
                if result_image_height > 0 and result_image_width > 0:
                    object_image = image[min_y:max_y, min_x:max_x]
                    result_segment = nx.dataset.utils.Segment(
                        label=result_oi_segment.label,
                        bbox=(0, 0, 1, 1),
                        additional_options=result_oi_segment.additional_options,
                    )

            if object_image is None:
                continue

            # Check result image with predictor:
            # if it predict some other object with conf >0.2 - ignore object,
            # this can be group or part of other object ...
            # We need to use only human confirmed objects with garantees that all objects on image labeled.
            object_image_h, object_image_w, _ = object_image.shape
            ignore_image = False
            predict_results = model.predict(object_image, conf=skip_confidence_threshold)
            result_file_name = oi_file.image_file.stem + "_" + str(uuid.uuid4())
            found_objects_with_class = []
            for res in predict_results:
                # TODO: Skip one label equal prediction with equal class and max iou and check other predictions.
                for cls_tensor, conf, xyxy in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
                    cls = round(cls_tensor.item())  # < expect numeric class here
                    conf = conf.item()
                    xyxy = xyxy.cpu().numpy()
                    norm_xyxy = [
                        xyxy[0] / object_image_w,
                        xyxy[1] / object_image_h,
                        xyxy[2] / object_image_w,
                        xyxy[3] / object_image_h
                    ]
                    if conf < skip_confidence_threshold or str(cls) not in ignore_result_images_with_classes:
                        continue
                    if str(cls) != str(result_segment.label):
                        ignore_image = True
                        break
                    else:
                        found_objects_with_class.append((norm_xyxy, conf))

            if ignore_image:
                logger.debug("Other object on object area predicted for " + str(image_id) + " - ignore object")

            # Check that here only one object of object class.
            if not ignore_image and len(found_objects_with_class) > 1:
                del found_objects_with_class[
                    max(enumerate(found_objects_with_class), key=lambda p: iou(p[1][0], [0, 0, 1, 1]))[0]
                ]
                if len(found_objects_with_class) > 0:
                    max_iou_el = max(found_objects_with_class, key=lambda p: iou(p[0], [0, 0, 1, 1]))
                    logger.debug(
                        "Few objects of target class predicted in object area for " +
                        str(image_id) + " - ignore object" +
                        ", iou = " + str(iou(max_iou_el[0], [0, 0, 1, 1])) +
                        ", conf = " + str(max_iou_el[1])
                    )
                    bboxes_dropped_by_prediction += 1
                    ignore_image = True

            if not ignore_image:
                # Divide objects to directories by size.
                result_h, result_w, _ = object_image.shape
                if (result_w <= TARGET_W and result_h <= TARGET_H):
                    # < For other sizes we can't create mosaic in <TARGET_W, TARGET_H> shape.
                    round_h = int(math.ceil(result_h / STEP_H) * STEP_H)
                    round_w = int(math.ceil(result_w / STEP_W) * STEP_W)
                    target_dir = pathlib.Path(target_root) / (str(round_w) + "x" + str(round_h))
                else:
                    target_dir = pathlib.Path(target_root) / "_"

                target_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target_dir / (result_file_name + ".jpg")), object_image)
                nx.dataset.utils.write_labelme_annotations(
                    str(pathlib.Path(target_dir) / (result_file_name + ".json")),
                    segments=[result_segment],
                    image_path=(result_file_name + ".jpg"),
                    image_width=result_w,
                    image_height=result_h,
                )

    logger.info(
        "From process files:\n" +
        "    bboxes_dropped_by_segments = " + str(bboxes_dropped_by_segments) + "\n" +
        "    bboxes_dropped_by_criteria = " + str(bboxes_dropped_by_criteria) + "\n" +
        "     bboxes_dropped_as_blocked = " + str(bboxes_dropped_as_blocked) + "\n" +
        "  bboxes_dropped_by_prediction = " + str(bboxes_dropped_by_prediction)
    )


if __name__ == "__main__":
    ultralytics.utils.LOGGER.setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)

    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(description='Util for convert Open Images annotations to labelme annotations.')
    parser.add_argument('--root', help = 'Open Images root', required = True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )
    parser.add_argument('--allow-grouped-objects', action='store_true')
    parser.add_argument(
        '--allow-non-verified', action='store_true',
        help='save only images that have object of required type with confidence=1 (manually processed)'
    )
    parser.add_argument('--model', help='model', type=str, required=True)
    parser.set_defaults(allow_grouped_objects=False, allow_non_verified=False)

    args = parser.parse_args()

    model = ultralytics.YOLO(args.model)
    open_images_convert(
        args.target_root,
        pathlib.Path(args.root),
        class_mapping={
            'Person': 0,
            'Bicycle': 1,
            'Car': 2,
            'Ambulance': 2,
            'Golf cart': 2,
            'Taxi': 2,
            'Snowmobile': 3,
            'Motorcycle': 3,
            'Airplane': 4,
            'Bus': 5,
            'Train': 6,
            'Truck': 7,
            'Boat': 8,
        },
        allow_grouped_objects=args.allow_grouped_objects,
        allow_non_verified=args.allow_non_verified,
        model=model,
    )
