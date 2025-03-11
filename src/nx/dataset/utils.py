import typing
import json
import copy
import collections
import numpy as np
import dataclasses
import pathlib
import logging
import shutil
import zlib
import base64
import pyclipper

import torch
import cv2
import ultralytics.data.augment


logger = logging.getLogger(__name__)


# Selected as min of dims for two humans on 000000000673.jpg (COCO set) (7x15, 7x17).
HARD_MIN_OBJECT_WIDTH_IN_PIXELS = 6
HARD_MIN_OBJECT_HEIGHT_IN_PIXELS = 6
# Combined both dimensions limit.
# Selected as min of dims for car on 000000003849.jpg (COCO set).
SOFT_MIN_OBJECT_WIDTH_IN_PIXELS = 11
SOFT_MIN_OBJECT_HEIGHT_IN_PIXELS = 11


class Segment(object):
    label: str
    polygon: typing.List[typing.Tuple[float, float]]
    bbox: typing.Tuple[float, float, float, float]  # xyxy format
    additional_options: typing.Dict = {}

    def __init__(self, label: str = None, polygon = None, bbox = None, additional_options = {}):
        self.label = str(label)
        self.polygon = polygon
        self.bbox = tuple(bbox) if bbox is not None else None
        self.additional_options = additional_options

    def __str__(self):
        return (
            "{ label = " + self.label + ", " +
            ("bbox = " + str(self.bbox) if self.bbox is not None else "polygon = " + str(self.polygon)) + "}"
        )

    def __eq__(self, other):
        return (
            self.label == other.label and
            (self.polygon is not None and other.polygon is not None and self.polygon == other.polygon) or
            (self.bbox is not None and other.bbox is not None and self.bbox == other.bbox)
        )

    def offset(self, offset_x = 0, offset_y = 0):
        res = copy.deepcopy(self)
        if res.bbox is not None:
            res.bbox = (
                max(res.bbox[0] + offset_x, 0),
                max(res.bbox[1] + offset_y, 0),
                min(res.bbox[2] + offset_x, 1),
                min(res.bbox[3] + offset_y, 1)
            )
        else:
            res.polygon = [
                (
                    min(max(point[0] + offset_x, 0), 1),
                    min(max(point[1] + offset_y, 0), 1)
                ) for point in res.polygon
            ]
        return res

    def iou(self, other):
        min_x, min_y, max_x, max_y = self.get_bbox()
        other_min_x, other_min_y, other_max_x, other_max_y = other.get_bbox()
        cross_min_x = max(min_x, other_min_x)
        cross_min_y = max(min_y, other_min_y)
        cross_max_x = min(max_x, other_max_x)
        cross_max_y = min(max_y, other_max_y)
        if cross_min_x >= cross_max_x:
            return 0
        if cross_min_y >= cross_max_y:
            return 0
        self_S = (max_x - min_x) * (max_y - min_y)
        other_S = (other_max_x - other_min_x) * (other_max_y - other_min_y)
        cross_S = (cross_max_x - cross_min_x) * (cross_max_y - cross_min_y)
        return cross_S / (self_S + other_S - cross_S)

    def get_bbox(self) -> typing.Tuple[float, float, float, float]:
        if self.bbox is not None:
            return self.bbox
        min_x = min(self.polygon, key=lambda x: x[0])[0]
        min_y = min(self.polygon, key=lambda x: x[1])[1]
        max_x = max(self.polygon, key=lambda x: x[0])[0]
        max_y = max(self.polygon, key=lambda x: x[1])[1]
        return (min_x, min_y, max_x, max_y)

    def convert_to_bbox_segment(self):
        return Segment(
            label=self.label,
            bbox=self.get_bbox(),
            additional_options=self.additional_options
        )

    def convert_to_polygon_segment(self):
        if self.polygon is not None:
            return self
        result = copy.deepcopy(self)
        result.polygon = self._get_polygon()
        result.bbox = None
        return result

    def _get_polygon(self):
        if self.polygon:
            return self.polygon
        x_left, y_top, x_right, y_bottom = self.bbox
        points = [
            [x_left, y_top],
            [x_right, y_top],
            [x_right, y_bottom],
            [x_left, y_bottom],
        ]
        return points


class SegmentsToBbox(ultralytics.data.augment.BaseTransform):
    """
    SegmentsToBbox: final augmentation(before standard ultralytics transformation) for fill bboxes by loadedSegments.
    """
    def __call__(self, labels):
        result = dict(labels)
        result['cls'], result['bboxes'] = segments_to_bboxes(result['loadedClasses'], result["loadedSegments"])
        return result


class FilterInvisibleBboxes(ultralytics.data.augment.BaseTransform):
    """
    FilterInvisibleBboxes
    """
    def __call__(self, label):
        if "bboxes" in label:
            image_shape = label['shape']  # HWC like in cv2.
            bboxes = label['bboxes']  # bboxes in YOLO format (center_x, center_y, w, h).
            classes = label['cls']
            segments = label['loaded_segments']
            result_bboxes = []
            result_classes = []
            result_segments = []
            for cls, bbox, segment in zip(classes, bboxes, segments):
                center_x = bbox[0]
                center_y = bbox[1]
                width = bbox[2]
                height = bbox[3]
                left_x = int(image_shape[1] * (center_x - width / 2))
                top_y = int(image_shape[0] * (center_y - height / 2))
                right_x = int(image_shape[1] * (center_x + width / 2))
                bottom_y = int(image_shape[0] * (center_y + height / 2))
                if (
                    (
                        right_x - left_x > HARD_MIN_OBJECT_WIDTH_IN_PIXELS and
                        bottom_y - top_y > HARD_MIN_OBJECT_HEIGHT_IN_PIXELS
                    ) and (
                        right_x - left_x > SOFT_MIN_OBJECT_WIDTH_IN_PIXELS or
                        bottom_y - top_y > SOFT_MIN_OBJECT_HEIGHT_IN_PIXELS
                    )
                ):
                    result_classes.append(cls)
                    result_bboxes.append(bbox)
                    result_segments.append(segment)
            if len(result_classes) != len(classes):
                label = copy.copy(label)
                label['cls'] = np.array(result_classes, dtype=np.float32)
                label['bboxes'] = torch.tensor(result_bboxes, dtype=np.float32)
                label['loaded_segments'] = torch.tensor(result_segments, dtype=np.float32)

        return label


def decode_zip_bitmap_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)

    imdecoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if (len(imdecoded.shape) == 3) and (imdecoded.shape[2] >= 4):
        mask = imdecoded[:, :, 3].astype(bool)  # < 4-channel imgs.
    elif len(imdecoded.shape) == 2:
        mask = imdecoded.astype(bool)  # < Flat 2d mask.
    else:
        raise RuntimeError('Wrong internal mask format.')
    return mask


def filter_segment(segment, width = 1, height = 1):
    min_x, min_y, max_x, max_y = segment.get_bbox()
    return (max_x - min_x) > 0.0001 and (max_y - min_y) > 0.0001 and (
        (
            (max_x - min_x) * width >= HARD_MIN_OBJECT_WIDTH_IN_PIXELS and
            (max_y - min_y) * height >= HARD_MIN_OBJECT_HEIGHT_IN_PIXELS
        ) and (
            (max_x - min_x) * width >= SOFT_MIN_OBJECT_WIDTH_IN_PIXELS or
            (max_y - min_y) * height >= SOFT_MIN_OBJECT_HEIGHT_IN_PIXELS
        )
    )


def segments_to_bboxes(
    segments: typing.List[Segment], image_width = None, image_height = None
) -> typing.Tuple[np.array, np.array, list]:
    yolo_bboxes = []
    result_classes = []
    filtered_segments = []
    for segment in segments:
        if filter_segment(segment, width=image_width, height=image_height):
            min_x, min_y, max_x, max_y = segment.get_bbox()
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            width = max_x - min_x
            height = max_y - min_y
            result_classes.append([float(segment.label)])
            yolo_bboxes.append([center_x, center_y, width, height])
            filtered_segments.append(segment)

    return (
        np.array(result_classes, dtype=np.float32) if result_classes else np.zeros((0, 1), dtype=np.float32),
        np.array(yolo_bboxes, dtype=np.float32) if yolo_bboxes else np.zeros((0, 4), dtype=np.float32),
        filtered_segments
    )


def read_labelme_annotations(labelme_file: str, string_labels = False):
    with open(labelme_file) as f:
        data = json.load(f)
    # parse shapes
    res = []
    for shape in data["shapes"]:
        label = str(shape["label"]) if string_labels else int(shape["label"])
        additional_options = {}
        if "is_truncated" in shape:
            additional_options['is_truncated'] = shape['is_truncated']
        if shape["shape_type"] == 'polygon':
            points = [
                [float(point[0]) / data['imageWidth'], float(point[1]) / data['imageHeight']]
                for point in shape["points"]
            ]
            result_segment = Segment(
                label=label, polygon=points, additional_options=additional_options
            )
        elif shape["shape_type"] == 'rectangle':
            x_left = float(shape["points"][0][0]) / data['imageWidth']
            y_top = float(shape["points"][0][1]) / data['imageHeight']
            x_right = float(shape["points"][1][0]) / data['imageWidth']
            y_bottom = float(shape["points"][1][1]) / data['imageHeight']
            result_segment = Segment(
                label=label, bbox=(x_left, y_top, x_right, y_bottom),
                additional_options=additional_options
            )
        else:
            raise Exception("read_labelme_annotations: unsupported shape_type = '" + str(shape["shape_type"]) + "'")

        res.append(result_segment)
    return (res, data['imageWidth'], data['imageHeight'])


def write_labelme_annotations(
    labelme_file: str,
    image_path: str = None,
    segments: typing.List[Segment] = [],
    image: np.ndarray = None,  # < cv2 image
    image_width: int = None,
    image_height: int = None,
):
    assert image is not None or (image_width is not None and image_height is not None)

    w = image.shape[1] if image is not None else image_width
    h = image.shape[0] if image is not None else image_height

    shapes = []
    for segment in segments:
        if segment.polygon is not None:
            obj = {
                "label": segment.label,  # < Save COCO class index as label.
                "group_id": None,
                "shape_type": "polygon",
                "points": [
                    [int(p[0] * w), int(p[1] * h)] for p in segment.polygon
                ]
            }
        else:
            assert segment.bbox is not None
            obj = {
                "label": segment.label,  # < Save COCO class index as label.
                "group_id": None,
                "shape_type": "rectangle",
                "points": [
                    [int(segment.bbox[0] * w), int(segment.bbox[1] * h)],
                    [int(segment.bbox[2] * w), int(segment.bbox[3] * h)],
                ]
            }
        obj.update(segment.additional_options)
        shapes.append(obj)

    labelme_json = {
        "version": "5.0.1",
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "flags": {},
        "imagePath": str(image_path),
        "imageData": None,
        "imageWidth": w,
        "imageHeight": h,
        "shapes": shapes
    }
    # Save lebelme markup to file.
    with open(labelme_file, 'w', encoding='utf-8') as f:
        json.dump(labelme_json, f, ensure_ascii=False, indent=4)


def min_distance_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_distance_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]: idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    f = (np.concatenate(s, axis=0)).reshape(-1).tolist()
    return [[f[p_index * 2], f[p_index * 2 + 1]] for p_index in range(int(len(f) / 2))]


@dataclasses.dataclass
class DatasetninjaAnnotatedFile(object):
    image_file: pathlib.Path
    annotations_file: pathlib.Path = None


def load_datasetninja_annotations(file_path, class_mapping):
    try:
        result_segments = []
        file_content = pathlib.Path(file_path).read_text()
        ann_json = json.loads(file_content)
        w = ann_json['size']['width']
        h = ann_json['size']['height']

        for obj in ann_json['objects']:
            if (
                'classTitle' in obj and
                obj['classTitle'] in class_mapping
            ):
                class_title = obj['classTitle']
                if 'geometryType' not in obj:
                    return None  # < Ignore file, we can't process other geometryType's.

                geometry_type = obj['geometryType']
                if geometry_type == 'polygon':
                    result_segments.append(Segment(
                        str(class_mapping[class_title]),
                        polygon=[(point[0] / w, point[1] / h) for point in obj['points']['exterior']],
                    ))
                elif geometry_type == 'rectangle':
                    bbox_points = obj['points']['exterior']
                    result_segments.append(Segment(
                        str(class_mapping[class_title]),
                        bbox=(
                            bbox_points[0][0] / w,
                            bbox_points[0][1] / h,
                            bbox_points[1][0] / w,
                            bbox_points[1][1] / h
                        ),
                    ))
                elif geometry_type == 'bitmap':
                    aligned_mask = decode_zip_bitmap_mask(obj['bitmap']['data'])
                    full_mask = np.zeros((h, w), dtype=bool)
                    aligned_mask_h, aligned_mask_w = aligned_mask.shape
                    paste_mask_x = obj['bitmap']['origin'][0]
                    paste_mask_y = obj['bitmap']['origin'][1]
                    full_mask[
                        paste_mask_y:paste_mask_y + aligned_mask_h,
                        paste_mask_x:paste_mask_x + aligned_mask_w
                    ] = aligned_mask
                    result_segment = mask_to_segment(full_mask)
                    result_segment.label = str(class_mapping[class_title])
                    if result_segment is not None:
                        result_segments.append(result_segment)
                else:
                    return None  # < Ignore file, we can't process other geometryType's.
        return result_segments
    except Exception as e:
        raise Exception("Error on reading '" + str(file_path) + "' annotations file") from e


def class_is_present(
    image: np.ndarray,  # < cv2 image.
    classes: dict,
    model = None,
    conf: float = 0.25
):
    if not classes:
        return False
    predict_results = model.predict(image, conf=conf)
    non_annotated_class_present: bool = False
    for res in predict_results:
        for cls_tensor, conf, xyxy in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
            cls = round(cls_tensor.item())
            if str(cls) in classes and conf > non_annotated_class_present:
                return str(cls)
    return False


def datasetninja_file_convert(
    target_root: str,
    datasetninja_file: DatasetninjaAnnotatedFile,
    model = None,
    class_mapping = None,
    ignore_files_with_classes = {},
    non_annotated_classes_conf_threshold = 0.25,
):
    if datasetninja_file.annotations_file is None:
        logger.debug("Ignore " + str(datasetninja_file.image_file) + " - no annotations file")
        return

    segments = load_datasetninja_annotations(datasetninja_file.annotations_file, class_mapping)
    if segments is None:
        logger.warning("Ignore " + str(datasetninja_file.image_file) + ": error on annotations reading")
        return

    image_file = datasetninja_file.image_file
    image = cv2.imread(str(image_file))
    h, w, _ = image.shape

    # Filter invisible segments (low size for example).
    filtered_segments = []
    for s in segments:
        if filter_segment(s, width=w, height=h):
            filtered_segments.append(s)
        else:
            logger.debug("Drop invisible segment on " + str(image_file))
    segments = filtered_segments

    # Check that image contains any required label.
    if segments is None or len(segments) == 0:
        logger.debug("Ignore " + str(image_file) + " - no objects after filtering")
        return

    if class_is_present(
        image,
        ignore_files_with_classes,
        model=model,
        conf=non_annotated_classes_conf_threshold
    ):
        logger.info("Ignore " + str(image_file) + " - contains non annotated class")
        return

    if image_file.suffix == '.jpg':
        shutil.copyfile(image_file, pathlib.Path(target_root) / (image_file.name))
    else:
        # Convert non jpg to jpg.
        cv2.imwrite(pathlib.Path(target_root) / (image_file.stem + ".jpg"), image)
        image_file = pathlib.Path(image_file.stem + ".jpg")

    # Save lebelme markup to file.
    result_labelme_file = str(pathlib.Path(target_root) / pathlib.Path(image_file.stem + ".json"))
    logger.debug(
        "Save annotations for '" + str(image_file.name) + "' to '" +
        result_labelme_file + "'"
    )
    write_labelme_annotations(
        result_labelme_file,
        segments=segments,
        image_path=image_file.name,
        image_width=w,
        image_height=h,
    )


def datasetninja_convert(
    target_root,  # < Directory for push result (jpeg + json in labelme format)
    images_root: typing.Union[str, pathlib.Path],
    annotations_root,
    model = None,
    class_mapping = {},
    ignore_files_with_classes = {},
    non_annotated_classes_conf_threshold = 0.25,
):
    datasetninja_files = {}
    images_root = pathlib.Path(images_root)

    for path in pathlib.Path(annotations_root).glob('*.json'):
        image_file = images_root / path.stem
        if image_file.is_file():
            datasetninja_files[path.stem] = DatasetninjaAnnotatedFile(
                annotations_file=path,
                image_file=image_file,
            )
        else:
            logger.error("Can't find image file for annotations file: " + str(path))

    for key, datasetninja_file in datasetninja_files.items():
        datasetninja_file_convert(
            target_root, datasetninja_file, model=model, class_mapping=class_mapping,
            ignore_files_with_classes=ignore_files_with_classes,
            non_annotated_classes_conf_threshold=non_annotated_classes_conf_threshold
        )


_PYCLIPPER_MUL = 1000


def _box_to_pyclipper_polygon(bbox):
    return (
        (int(bbox[0] * _PYCLIPPER_MUL), int(bbox[1] * _PYCLIPPER_MUL)),
        (int(bbox[2] * _PYCLIPPER_MUL), int(bbox[1] * _PYCLIPPER_MUL)),
        (int(bbox[2] * _PYCLIPPER_MUL), int(bbox[3] * _PYCLIPPER_MUL)),
        (int(bbox[0] * _PYCLIPPER_MUL), int(bbox[3] * _PYCLIPPER_MUL)),
    )


def blocked_area_percentage(
    segment: Segment,
    block_segments: typing.List[Segment]
) -> float:
    if len(block_segments) == 0:
        return 0

    try:
        pc = pyclipper.Pyclipper()
        if segment.polygon is not None:
            pyclipper_segment = tuple([[int(p[0] * _PYCLIPPER_MUL), int(p[1] * _PYCLIPPER_MUL)] for p in segment.polygon])
        else:
            pyclipper_segment = _box_to_pyclipper_polygon(segment.bbox)

        segment_area = pyclipper.Area(pyclipper_segment)
        if segment_area < 0.0000001:
            return 0
        pc.AddPath(pyclipper_segment, pyclipper.PT_CLIP, True)

        block_polygons = []
        for block_segment in block_segments:
            if block_segment.polygon is not None:
                block_polygons.append(
                    tuple([(int(p[0] * _PYCLIPPER_MUL), int(p[1] * _PYCLIPPER_MUL)) for p in block_segment.polygon])
                )
            else:
                block_polygons.append(_box_to_pyclipper_polygon(block_segment.bbox))

        pc.AddPaths(tuple(block_polygons), pyclipper.PT_SUBJECT, True)
        blocked_polygons = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)

        blocked_area = 0
        for blocked_polygon in blocked_polygons:
            blocked_area += abs(pyclipper.Area(tuple(blocked_polygon)))

        return blocked_area / segment_area
    except pyclipper.ClipperException as e:
        raise RuntimeError(str(e)) from e


def find_blocked_segments(check_segments, block_segments, cross_threshold = 0.25):
    """
    Find segments blocked by block_segments more then for cross_threshold,
    and all other segments that blocked by already blocked segments.
    """
    block_segments = list(block_segments)
    new_blocked_segments = block_segments
    while len(new_blocked_segments) > 0:
        new_check_segments = []
        new_blocked_segments = []
        for check_segment in check_segments:
            if blocked_area_percentage(check_segment, block_segments) > cross_threshold:
                new_blocked_segments.append(check_segment)
            else:
                new_check_segments.append(check_segment)
        block_segments += new_blocked_segments
        check_segments = new_check_segments
    return check_segments, block_segments


def mask_to_segment(mask: np.ndarray):
    if mask.dtype == bool:
        mask = mask.astype(dtype=np.uint8)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_SIMPLE)
    object_segments = []
    class_index = 0
    for contour in contours:
        contour = contour.squeeze()
        if len(contour) > 2:
            object_segments.append([[p[0] / w, p[1] / h] for p in contour])

    if len(object_segments) > 1:
        return Segment(
            label=str(class_index),
            polygon=merge_multi_segment(object_segments)
        )
    elif len(object_segments) > 0:
        return Segment(
            label=str(class_index),
            polygon=object_segments[0]
        )
    return None


def normalize_class_name(name: str):
    return name.strip().lower()


def coco_convert(  # noqa: C901
    target_dir,
    coco_json_dir,  # < Directory with *.json annotations in COCO format
    images_root,
    no_labels_dir = None,
    class_mapping = {},
    model = None,
    ignore_files_with_classes = {},
    non_annotated_classes_conf_threshold = 0.25,
):
    class_mapping = copy.copy(class_mapping)
    target_dir = pathlib.Path(target_dir)
    no_labels_dir = pathlib.Path(no_labels_dir) if no_labels_dir is not None else None

    # Collect all image names to full path mapping.
    image_paths = {}
    for path in pathlib.Path(images_root).rglob('*.jpg'):
        print("Process image file: " + str(path))
        image_paths[path.name] = path

    # Converts COCO JSON format to LabelMe format (use segments annotations).
    images = {}  # < imageId => imageInfo.
    image_coco_annotations = collections.defaultdict(list)  # < imageId => annotations.

    for json_file in sorted(pathlib.Path(coco_json_dir).rglob("*.json")):
        print("Process file: " + str(json_file))
        logger.info("Process file: " + str(json_file))
        with open(json_file) as f:
            data = json.load(f)

        # Add images to image dict.
        images.update({"{:g}".format(x["id"]): x for x in data["images"]})

        # If class mapping contains names - add numeric category id additionally.
        if "categories" in data:
            for category in data["categories"]:
                if "id" in category and "name" in category:
                    norm_class_name = normalize_class_name(category["name"])
                    if norm_class_name in class_mapping:
                        norm_class_name = normalize_class_name(category["name"])
                        class_mapping[str(category["id"])] = class_mapping[norm_class_name]

        if "annotations" in data:
            for ann in data["annotations"]:
                image_coco_annotations[ann["image_id"]].append(ann)

    # Save image and annotations for files with annotations and existing images.
    for image_id, annotations in image_coco_annotations.items():
        if str(image_id) not in images:
            continue

        # Find full image path for save annotations next to it.
        # Parse COCO annotations.
        img = images[f"{image_id:g}"]
        h, w, file_name = img["height"], img["width"], img["file_name"]
        file_name = pathlib.Path(file_name).name

        if file_name not in image_paths:
            #logger.info("Annotations for " + file_name + " skipped - can't find image file")
            continue

        segments = []
        for ann in annotations:
            if "iscrowd" in ann and ann["iscrowd"]:
                continue
            if "category_id" not in ann:
                # Ignore image annotations.
                continue
            category_id = str(ann["category_id"])

            # Convert COCO label to Nx.
            nx_label = class_mapping[category_id] if category_id in class_mapping else None
            if nx_label is not None:
                if 'segmentation' in ann and len(ann['segmentation']) > 0:
                    # Use only segments for train after with augmentations (rotate, ...).
                    if len(ann["segmentation"]) > 1:
                        s = nx.dataset.utils.merge_multi_segment(ann["segmentation"])
                    else:
                        s = [j for i in ann["segmentation"] for j in i]  # All segments concatenated.
                    polygon = [ [p[0] / w, p[1] / h] for p in (np.array(s).reshape(-1, 2)).tolist() ]
                    segment = Segment(
                        polygon=polygon,
                        label=str(nx_label)
                    )
                    if segment not in segments:  # < Deduplicate.
                        segments.append(segment)
                elif 'bbox' in ann and len(ann['bbox']) >= 4:
                    source_bbox = ann['bbox']
                    segment = Segment(
                        label=str(nx_label),
                        bbox=[  # < coco bbox have xywh format.
                            source_bbox[0] / w,
                            source_bbox[1] / h,
                            (source_bbox[0] + source_bbox[2]) / w,
                            (source_bbox[1] + source_bbox[3]) / h
                        ],
                    )
                    if segment not in segments:  # < Deduplicate.
                        segments.append(segment)

        # Remove bboxes annotated as segments.
        result_bboxes = []
        for segment_i, segment in enumerate(segments):
            if segment.bbox is not None:
                drop_bbox = False
                for check_segment in segments:
                    if check_segment.polygon is not None:
                        iou = segment.iou(check_segment)
                        assert iou >= 0 and iou <= 1
                        if (
                            segment.label == check_segment.label and
                            iou > 0.98
                        ):
                            drop_bbox = True
                            break
                if not drop_bbox:
                    result_bboxes.append(segment)

        segments = result_bboxes + [s for s in segments if s.polygon is not None]

        image_file_path = image_paths[file_name]
        result_file_name = pathlib.Path(image_file_path.stem[:80] + image_file_path.suffix)

        orig_segments = copy.deepcopy(segments)
        segments = [s for s in segments if filter_segment(s, width=w, height=h)]

        if len(segments) > 0:
            if ignore_files_with_classes:
                image = cv2.imread(str(image_file_path))
                non_annotated_class = class_is_present(
                    image,
                    ignore_files_with_classes,
                    model=model,
                    conf=non_annotated_classes_conf_threshold
                )
                if non_annotated_class:
                    logger.info(
                        "Ignore " + str(image_file_path.name) + " - contains non annotated class: " +
                        str(non_annotated_class)
                    )
                    continue

            shutil.copy(
                str(image_file_path),
                str(target_dir / result_file_name.name)
            )
            write_labelme_annotations(
                str(target_dir / pathlib.Path(result_file_name.stem + ".json")),
                segments=segments,
                image_path=result_file_name,
                image_width=w,
                image_height=h,
            )
        elif no_labels_dir is not None:
            shutil.copy(
                str(image_file_path),
                str(no_labels_dir / result_file_name.name)
            )
        else:
            logger.debug("Skip image without objects: " + image_file_path.name)


def adapt_image_size(image, max_result_width = 640, max_result_height = 640):
    h, w, _ = image.shape
    modified = False
    if w > max_result_width or h > max_result_height:
        resize_ratio = min(max_result_width / w, max_result_height / h)
        result_width = int(w * resize_ratio)
        result_height = int(h * resize_ratio)
        image = cv2.resize(image, (result_width, result_height))
        modified = True
    return image, modified
