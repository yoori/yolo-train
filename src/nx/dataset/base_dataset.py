import os
import enum
import pathlib
import logging
import json
from dataclasses import dataclass

import torch
import ultralytics.data.dataset

import nx.dataset.utils


logger = logging.getLogger(__name__)


# Fixed version of collate_fn (from ultralytics/data/dataset.py),
# original implementation have bug - it expect that dict.values keep order of fields and this order is equal to .keys().
def collate_fn(batch):
    new_batch = {}
    keys = batch[0].keys()
    values = list(zip(*[[b[k] for k in keys] for b in batch]))
    for i, k in enumerate(keys):
        try:
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
        except Exception as e:
            raise Exception("Collate failed on '" + str(k) + "'") from e
        new_batch[k] = value
    new_batch["batch_idx"] = list(new_batch["batch_idx"])
    for i in range(len(new_batch["batch_idx"])):
        new_batch["batch_idx"][i] += i  # add target image index for build_targets()
    new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
    return new_batch


class BaseDataset(ultralytics.data.dataset.YOLODataset):
    @dataclass
    class DatasetPartConfig(object):
        priority: int = 1
        noise: bool = True
        mosaic: bool = True

    class IsolateType(enum.Enum):
        NOT_ISOLATED = 0
        SEGMENT_ISOLATED = 1
        BBOX_ISOLATED = 2

    # Override get_labels for load labelme markup files.
    _labels = None
    backgrounds = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_labels(self):
        if self._labels is not None:
            return self._labels

        label_configs = {}

        # fetch self.im_files and find .json for each.
        logger.info("To load labels")
        result_labels = []
        result_im_files = []
        result_label_files = []

        for image_file_path_str in self.im_files:
            image_file_path = pathlib.Path(image_file_path_str)
            if not str(image_file_path.parent) in label_configs:
                # Load dataset part config.
                part_config_file = image_file_path.parent / pathlib.Path('config.json')
                if os.path.isfile(part_config_file):
                    with open(part_config_file) as f:
                        data = json.load(f)
                        label_configs[str(image_file_path.parent)] = BaseDataset.DatasetPartConfig(
                            priority=data.get('priority', 1),
                            noise=data.get('noise', True),
                            mosaic=data.get('mosaic', True),
                        )
                else:
                    label_configs[str(image_file_path.parent)] = BaseDataset.DatasetPartConfig()

            label_config = label_configs[str(image_file_path.parent)]
            labelme_file = str(image_file_path.parent / pathlib.Path(image_file_path.stem + ".json"))
            if os.path.isfile(labelme_file):
                segments, image_width, image_height = nx.dataset.utils.read_labelme_annotations(labelme_file)

                can_apply_only_rect_safe_transformations = False
                for segment_i, segment in enumerate(segments):
                    if segment.polygon is None:
                        segments[segment_i] = segment.convert_to_polygon_segment()
                        can_apply_only_rect_safe_transformations = True

                if can_apply_only_rect_safe_transformations:
                    logger.debug("For " + str(labelme_file) + " is available only rect safe transformations")

                cls, bboxes, segments = nx.dataset.utils.segments_to_bboxes(
                    segments,
                    image_width=image_width,
                    image_height=image_height
                )

                assert len(cls) == len(bboxes) and len(cls) == len(segments), (
                    "len(cls) = " + str(len(cls)) + ", len(bboxes) = " + str(len(bboxes)))

                # Collect isolated objects - object for that bbox don't cross other bboxes.
                segments_isolations = []
                for segment in segments:
                    try:
                        if (nx.dataset.utils.blocked_area_percentage(
                            segment.convert_to_bbox_segment(),
                            [s.convert_to_bbox_segment() for s in segments if s != segment]
                        ) < 0.001):
                            segments_isolations.append(BaseDataset.IsolateType.BBOX_ISOLATED)
                        elif (nx.dataset.utils.blocked_area_percentage(
                            segment,
                            [s for s in segments if s != segment]
                        ) < 0.001):
                            segments_isolations.append(BaseDataset.IsolateType.SEGMENT_ISOLATED)
                        else:
                            segments_isolations.append(BaseDataset.IsolateType.NOT_ISOLATED)
                    except RuntimeError as e:
                        logger.error("Exception: " + str(type(e)))
                        segments_isolations.append(BaseDataset.IsolateType.NOT_ISOLATED)
                new_label = {
                    'im_file': str(image_file_path),
                    'shape': (image_height, image_width),
                    'cls': cls,
                    'bboxes': bboxes,
                    # On trasformation bboxes will be refilled by loaded_segments: +bboxes, +bbox_format="xywh".
                    'loaded_segments': segments,
                    'can_apply_only_rect_safe_transformations': can_apply_only_rect_safe_transformations,
                    'normalized': True,
                    'bbox_format': 'xywh',  # < return bboxes in yolo format: [center_x, center_y, w, h].
                }
                if len(segments) > 0:
                    mask_file_path = str(image_file_path.parent / pathlib.Path(image_file_path.stem + "_mask.png"))
                    # Set of field names should be equal for elements (required in collate_fn),
                    # Fill fields with None if here no value.
                    new_label['mask_file'] = mask_file_path if os.path.isfile(mask_file_path) else None
                    for i in range(label_config.priority):
                        result_im_files.append(image_file_path_str)
                        result_label_files.append(labelme_file)
                        result_labels.append(new_label)
                else:
                    self.backgrounds.append(new_label)

        logger.info(
            "From load labels: " + str(len(result_labels)) + " items loaded, " +
            str(len(self.backgrounds)) + " backgrounds loaded"
        )

        self.im_files = result_im_files
        self.label_files = result_label_files
        self._labels = result_labels

        return result_labels
