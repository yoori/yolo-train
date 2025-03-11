import os
import typing
import numpy as np
import logging
import cv2
import torch
import copy
import random
import tempfile
import time
import gc
import datetime

import nvidia.dali as dali
import nvidia.dali.plugin.pytorch as dali_plugin_pytorch

import nx.dataset.utils
import nx.dataset.mosaic_utils
import nx.dataset.base_dataset


logger = logging.getLogger(__name__)


# DALI don't have normal interface for apply geomethric transformations to image and keypoints.
# make
def paste(
    images,
    keypoints,
    ratio = None,  # < ratio for resize image.
    paste_x = None,
    paste_y = None
):
    images = dali.fn.paste(
        images,
        fill_value=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ),
        # < Paste random color with equal components.
        paste_x=paste_x,
        paste_y=paste_y,
        ratio=ratio,
    )
    keypoints = (keypoints + (paste_x * [1., 0] + paste_y * [0, 1.]) * (ratio - 1.0)) / ratio
    return images, keypoints


def rotate(images, keypoints, angles=0):
    image_wh = images.shape()[1::-1]  # get WH from HWC shape
    # dali.fn.rotate and dali.fn.transforms.rotation rotate in different directions.
    rotation_matrixes = dali.fn.transforms.rotation(angle=-angles)
    images = dali.fn.rotate(images, angle=angles, fill_value=0)
    rotated_image_wh = images.shape()[1::-1]

    keypoints = dali.fn.coord_transform(
        image_wh * (keypoints - [0.5, 0.5]), MT=rotation_matrixes
    ) / rotated_image_wh + (0.5, 0.5)

    return images, keypoints


def resize_and_crop(images, keypoints = None, size = 640):
    image_wh = images.shape()[1::-1]  # get WH from HWC shape
    images = dali.fn.resize(images, resize_longer=size)
    images = dali.fn.crop(images, crop=(size, size), out_of_bounds_policy="pad", fill_values=0)
    if keypoints is not None:
        mul = dali.math.min(image_wh / image_wh[::-1], [1., 1.])  # (min(w / h, 1), min(h / w, 1))
        keypoints = (keypoints - [0.5, 0.5]) * mul + [0.5, 0.5]
    return images, keypoints


def resize_and_crop_without_keypoints(images, size = 640):
    images = dali.fn.resize(images, resize_longer=size)
    images = dali.fn.crop(images, crop=(size, size), out_of_bounds_policy="pad", fill_values=0)
    return images


def hflip(images, keypoints, probability = 0.5):
    do_hflip = dali.fn.random.coin_flip(probability=probability, device='cpu')
    if do_hflip:
        images = dali.fn.flip(images, device='gpu', horizontal=1, vertical=0)
        keypoints = (keypoints - [0.5, 0.0]) * [-1, 1] + [0.5, 0.0]
    return images, keypoints


def vflip(images, keypoints, probability = 0.5):
    do_hflip = dali.fn.random.coin_flip(probability=probability, device='cpu')
    if do_hflip:
        images = dali.fn.flip(images, device='gpu', horizontal=0, vertical=1)
        keypoints = (keypoints - [0.0, 0.5]) * [1, -1] + [0.0, 0.5]
    return images, keypoints


def background_paste(images, mask_images, background_images, keypoints=None, size=640):
    images, keypoints = resize_and_crop(images, keypoints=keypoints, size=size)
    mask_images = resize_and_crop_without_keypoints(mask_images, size=size)
    background_images = dali.fn.resize(background_images, size=(size, size))
    images = (
        (images & mask_images) |  # Cutted objects.
        dali.fn.cast(
            background_images & (
                - mask_images + dali.fn.ones(shape=(3), dtype=dali.types.DALIDataType.UINT8).gpu() * 255
            ),
            dtype=dali.types.DALIDataType.UINT8
        )  # < Background around objects.
    )
    return images, keypoints


def background_paste_2x2(
    images,  # < TensorList of images for transform.
    mask_images,  # < TensorList of masks that cover all objects on images.
    background_images_arr,  # < TensorList of background images for paste cutted objects.
    # 4 images.
    keypoints,  # < TensorList of objects keypoints (segments) for images.
    size=640  # < Size for unify images sizes before paste to backgrounds.
):
    images, keypoints = resize_and_crop(images, keypoints=keypoints, size=size)
    mask_images, _ = resize_and_crop(mask_images, size=size)
    image_cells = [[None, None], [None, None]]
    resized_images = dali.fn.resize(images, resize_x=(size // 2), resize_y=(size // 2))
    resized_mask_images = dali.fn.resize(mask_images, resize_x=(size // 2), resize_y=(size // 2))
    # Select random position for original image. It should be equal for batch, but it isn't problem.
    orig_x = random.randint(0, 1)
    orig_y = random.randint(0, 1)
    for x in range(2):
        for y in range(2):
            if x == orig_x and y == orig_y:
                # Paste original images.
                image_cells[x][y] = resized_images
            else:
                local_background_images = dali.fn.crop(
                    background_images_arr[y * 2 + x],
                    crop_pos_x=0.5 * x,
                    crop_pos_y=0.5 * y,
                    crop_w=(size // 2),
                    crop_h=(size // 2)
                )
                image_cells[x][y] = (
                    (resized_images & resized_mask_images) |  # Cutted objects.
                    dali.fn.cast(
                        local_background_images & (
                            - resized_mask_images +
                            dali.fn.ones(shape=(3), dtype=dali.types.DALIDataType.UINT8).gpu() * 255
                        ),
                        dtype=dali.types.DALIDataType.UINT8
                    )  # < Background around objects.
                )
    lines0 = dali.fn.cat(image_cells[0][0], image_cells[1][0], axis=0)
    lines1 = dali.fn.cat(image_cells[0][1], image_cells[1][1], axis=0)
    images = dali.fn.cat(lines0, lines1, axis=1)

    keypoints0 = dali.fn.cat(keypoints / 2, keypoints / 2 + [0.5, 0])
    keypoints1 = dali.fn.cat(keypoints / 2 + [0, 0.5], keypoints / 2 + [0.5, 0.5])
    keypoints = dali.fn.cat(keypoints0, keypoints1)

    return images, keypoints


def get_rel_start_by_bboxes(bboxes):
    return [np.float32((bbox[0], bbox[1])) for bbox in bboxes]


def get_rel_end_by_bboxes(bboxes):
    return [np.float32((bbox[2], bbox[3])) for bbox in bboxes]


def get_data(keypoints):
    return [np.float32(x) for x in keypoints]


def get_filename_data(files):
    return files


def random_grayscale(images, probability = 1.0):
    do_grayscale = dali.fn.random.coin_flip(
        probability=probability, dtype=dali.types.DALIDataType.BOOL
    )
    if do_grayscale:
        images = dali.fn.color_space_conversion(
            images, image_type=dali.types.RGB, output_type=dali.types.GRAY
        )
        images = dali.fn.color_space_conversion(
            images, image_type=dali.types.GRAY, output_type=dali.types.RGB
        )
    return images


def random_hsv(images, probability = 1.0):
    do_hsv = dali.fn.random.coin_flip(
        probability=probability, dtype=dali.types.DALIDataType.BOOL
    )
    if do_hsv:
        images = dali.fn.hsv(
            images,
            hue=dali.fn.random.uniform(range=(0, 360)),
            saturation=dali.fn.random.uniform(range=(0.9, 1.1))
        )
    return images


def get_temp_image_name():
    full_mask_file_holder = tempfile.NamedTemporaryFile(suffix='.jpg')
    full_mask_file = full_mask_file_holder.__enter__()
    file_name = full_mask_file.name
    full_mask_file.__exit__(None, None, None)
    return file_name


def apply_color_space_conversions(images, augment_args = {}):
    images = random_grayscale(images, probability=augment_args.get('to_gray', 0.1))

    images = random_hsv(images, probability=augment_args.get('hsv', 0.05))

    do_bgr = dali.fn.random.coin_flip(
        probability=augment_args.get('bgr', 0.1), dtype=dali.types.DALIDataType.BOOL)
    if do_bgr:
        images = dali.fn.color_space_conversion(
            images, image_type=dali.types.RGB, output_type=dali.types.BGR
        )

    do_color_twist = dali.fn.random.coin_flip(
        probability=augment_args.get('color_twist', 0.05), dtype=dali.types.DALIDataType.BOOL)
    if do_color_twist:
        images = dali.fn.color_twist(
            images,
            brightness=dali.fn.random.uniform(range=(0.8, 1.2)),
            contrast=dali.fn.random.uniform(range=(0.8, 1.2)),
            saturation=dali.fn.random.uniform(range=(0.8, 1.2)),
            hue=dali.fn.random.uniform(range=(-0.5, 0.5)),
        )

    return images


def apply_noise_conversions(images, augment_args = {}):
    do_water = dali.fn.random.coin_flip(
        probability=augment_args.get('water', 0.5), dtype=dali.types.DALIDataType.BOOL)
    if do_water:
        images = dali.fn.water(
            images,
            ampl_x=2,
            ampl_y=2,
            freq_x=0.1,
            freq_y=0.1,
        )

    do_jpeg_artifacts = dali.fn.random.coin_flip(
        probability=augment_args.get('jpeg', 0.05), dtype=dali.types.DALIDataType.BOOL)
    if do_jpeg_artifacts:
        images = dali.fn.jpeg_compression_distortion(images, quality=30)

    do_gaussian_blur = dali.fn.random.coin_flip(
        probability=augment_args.get('gaussian_blur', 0.05), dtype=dali.types.DALIDataType.BOOL)
    if do_gaussian_blur:
        images = dali.fn.gaussian_blur(
            images,
            window_size=dali.fn.random.uniform(range=(0, 4), dtype=dali.types.DALIDataType.INT32) * 2 + 1
        )

    do_noise_gaussian = dali.fn.random.coin_flip(
        probability=augment_args.get('gaussian_noise', 0.05), dtype=dali.types.DALIDataType.BOOL)
    if do_noise_gaussian:
        images = dali.fn.noise.gaussian(
            images,
            mean=dali.fn.random.uniform(range=(0., 0.4)),
            stddev=dali.fn.random.uniform(range=(0.05, 0.15))
        )

    do_salt_and_pepper = dali.fn.random.coin_flip(
        probability=augment_args.get('salt_and_pepper', 0.1), dtype=dali.types.DALIDataType.BOOL)
    if do_salt_and_pepper:
        images = dali.fn.noise.salt_and_pepper(
            images,
            prob=dali.fn.random.uniform(range=(0., 0.1))
        )

    do_grid_mask = dali.fn.random.coin_flip(
        probability=augment_args.get('grid_mask', 0.1), dtype=dali.types.DALIDataType.BOOL)
    if do_grid_mask:
        images = dali.fn.grid_mask(images, ratio=0.25)

    return images


# cut_regions: regions of images that we process (keypoints should be defined relative these regions)
@dali.pipeline_def(enable_conditionals=True, exec_dynamic=True)
def dali_load_and_augment_pipeline(
    files: typing.List[str] = None,
    mask_files: typing.List[str] = None,
    keypoints: typing.List[typing.List[typing.Tuple[float, float]]] = None,
    cut_regions: typing.List[typing.Tuple[float, float, float, float]] = None,
    background_files = None,
    empty_background = None,  # < file path for use as background when it isn't needs.
    return_kornia_format = False,
    apply_only_rect_safe_transformations = False,
    augment_args: typing.Dict = {},
    manual_garbage_collection = False
):
    result_size = 640

    cut_rel_start = dali.fn.external_source(
        source=lambda: get_rel_start_by_bboxes(cut_regions),
        dtype=dali.types.FLOAT
    )
    cut_rel_end = dali.fn.external_source(
        source=lambda: get_rel_end_by_bboxes(cut_regions),
        dtype=dali.types.FLOAT
    )

    process_keypoints = dali.fn.external_source(
        source=lambda: get_data(keypoints),
        dtype=dali.types.FLOAT
    )
    files, labels = dali.fn.readers.file(
        files = files,
        labels = list(range(len(files)))
    )
    images = dali.fn.decoders.image(
        files,
        device="mixed",
        output_type=dali.types.DALIImageType.RGB  # < By fact, output will be BGR
    )

    images = dali.fn.slice(
        images,
        rel_start=cut_rel_start,
        rel_end=cut_rel_end,
        out_of_bounds_policy='trim_to_shape',  # < avoid pixel out of bound for float defined start and end.
    )

    # Make backround paste transformations before all other augmentations (
    # mask should have correct position for apply it).
    if mask_files and background_files:
        mask_jpegs, mask_labels = dali.fn.readers.file(
            files = mask_files,
            labels = list(range(len(mask_files))),
        )
        mask_images = dali.fn.decoders.image(
            mask_jpegs,
            device="mixed",
            output_type=dali.types.DALIImageType.BGR  # < By fact, output will be RGB
        )
        mask_images = dali.fn.slice(
            mask_images,
            rel_start=cut_rel_start,
            rel_end=cut_rel_end,
            out_of_bounds_policy='trim_to_shape',  # < avoid pixel out of bound for float defined start and end.
        )

        # Choose images for apply background transformation - for other read 1x1 as background
        # Fill array of background image arrays with shuffled order.
        do_background_2x2 = dali.fn.random.coin_flip(
            probability=augment_args.get('replace_background_2x2', 0.25),
            device='cpu',
            dtype=dali.types.DALIDataType.BOOL
        )
        background_jpegs_arr = []
        background_images_arr = []
        shuffled_background_files_arr = []  # < array of 4 arrays with background images.

        # Fill background files outside of read loop.
        # DALI have specific problems with compile 'if DataNode' inside loop.
        if do_background_2x2:
            for i in range(4):
                shuffled_background_files = list(background_files)
                random.shuffle(shuffled_background_files)
                shuffled_background_files_arr.append(shuffled_background_files)
        else:
            for i in range(4):
                shuffled_background_files_arr.append([ empty_background ] * len(background_files))

        for i in range(4):
            shuffled_background_files = shuffled_background_files_arr[i]
            background_jpegs, background_labels = dali.fn.readers.file(
                files=shuffled_background_files,
                labels=list(range(len(shuffled_background_files))),
            )
            background_images = dali.fn.decoders.image(
                background_jpegs,
                device="mixed",
                output_type=dali.types.DALIImageType.BGR  # < By fact, output will be RGB
            )
            background_jpegs_arr.append(background_jpegs)
            background_images_arr.append(background_images)

        if do_background_2x2:
            images, process_keypoints = background_paste_2x2(
                images, mask_images, background_images_arr, keypoints=process_keypoints
            )
        else:
            do_background = dali.fn.random.coin_flip(
                probability=augment_args.get('replace_background', 0.3),
                device='cpu',
                dtype=dali.types.DALIDataType.BOOL
            )
            if do_background:
                images, process_keypoints = background_paste(
                    images, mask_images, background_images_arr[0], keypoints=process_keypoints
                )

    # Color space conversions.
    images = apply_color_space_conversions(images, augment_args=augment_args)

    if manual_garbage_collection:
        gc.collect()

    # Transformations without coordinates change.
    images = apply_noise_conversions(images, augment_args=augment_args)

    if manual_garbage_collection:
        gc.collect()

    if not apply_only_rect_safe_transformations:
        # Transformations with coordinates change.
        # rotate before pad for keep image inside viewport fully.
        do_rotate = dali.fn.random.coin_flip(
            probability=augment_args.get('rotate', 0.2), dtype=dali.types.DALIDataType.BOOL)
        if do_rotate:
            angles = dali.fn.random.uniform(range=(-45, 45), device='cpu')
            images, process_keypoints = rotate(images, process_keypoints, angles=angles)

    do_paste = dali.fn.random.coin_flip(
        probability=augment_args.get('paste', 0.05), dtype=dali.types.DALIDataType.BOOL)
    if do_paste:
        images, process_keypoints = paste(
            images,
            process_keypoints,
            ratio=dali.fn.random.uniform(range=(1.0, 2.0), device='cpu'),
            paste_x=dali.fn.random.uniform(range=(0.0, 1.0), device='cpu'),
            paste_y=dali.fn.random.uniform(range=(0.0, 1.0), device='cpu'),
        )

    if manual_garbage_collection:
        gc.collect()

    images, process_keypoints = hflip(
        images, process_keypoints, probability=augment_args.get('hflip', 0.3)
    )
    images, process_keypoints = vflip(
        images, process_keypoints, probability=augment_args.get('vflip', 0.05)
    )

    if manual_garbage_collection:
        gc.collect()

    # normalize for equal sizes (for batch processing).
    images, process_keypoints = resize_and_crop(images, process_keypoints, size=result_size)

    # Adapt image format.
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    if return_kornia_format:
        images = dali.fn.cast(images, device='gpu', dtype=dali.types.DALIDataType.FLOAT)
    else:
        images = dali.fn.cast(images, dtype=dali.types.DALIDataType.UINT8)

    return images, labels, process_keypoints


DALI_DEFAULT_AUGMENT_ARGS = {
    "replace_background_2x2": 0.25,
    "replace_background": 0.3,
    "to_gray": 0.1,
    "hsv": 0.05,
    "bgr": 0.1,
    "water": 0.05,
    "jpeg": 0.05,
    "color_twist": 0.05,
    "gaussian_blur": 0.05,
    "gaussian_noise": 0.05,
    "salt_and_pepper": 0.1,
    "grid_mask": 0.1,
    "rotate": 0.2,
    "paste": 0.05,
    "hflip": 0.3,
    "vflip": 0.05,
    'mixup3': 0.1,
    'mixup2': 0.2,
    'object_mosaic': 0.1,
    'object_overlay_mosaic': 0.1,  # < mosaic with self overlay.
}


class DaliKorniaDataset(nx.dataset.base_dataset.BaseDataset):
    """
    DaliKorniaDataset:
      * Override get_labels for load labelme markup files.
      * Customize augmentations by override build_transforms: it should convert segments to bboxes.
      * use nvidia.dali augmentations instead CPU implemenations.
    augment_args default:
    """
    _cut_segments_for_labels: typing.Dict
    _corner_keypoints = None

    def __init__(self, *args, imgsz=640, augment=False, augment_args=None, **kwargs):
        self._imgsz = imgsz
        self._augment = augment
        self._augment_args = dict(DALI_DEFAULT_AUGMENT_ARGS)
        if augment_args is not None:
            self._augment_args.update(augment_args)
        kwargs['augment'] = False
        super().__init__(*args, imgsz=imgsz, **kwargs)
        if augment:
            self._full_mask_file = get_temp_image_name()
            full_mask_image = np.zeros((imgsz, imgsz, 3), np.uint8)
            full_mask_image[:] = (255, 255, 255)
            cv2.imwrite(self._full_mask_file, full_mask_image)
        self._cut_segments_for_labels = {}
        self._corner_keypoints = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self._trace_steps = False
        self._empty_background_file = get_temp_image_name()
        empty_background_image = np.zeros((1, 1, 3), np.uint8)
        empty_background_image[:] = (255, 255, 255)
        cv2.imwrite(self._empty_background_file, empty_background_image)

    def build_transforms(self, hyp=None):
        if self._augment:
            return nx.dataset.utils.FilterInvisibleBboxes()  # Disable ultralytics augmentations.
        # Use ultralytics.data.dataset.YOLODataset transforms (letterbox) if here no augmentations.
        return super().build_transforms(hyp=hyp)

    def get_labels(self):
        labels = super(DaliKorniaDataset, self).get_labels()

        if self._augment:
            for label in labels:
                # Don't keep data as torch.tensors - number of tensors for app is limited.
                label['normalized'] = True
                label['bbox_format'] = 'xywh'
                label['batch_idx'] = np.array([0] * len(label['cls']), dtype=np.float32)

        return labels

    def __getitem__(self, index):
        res = self.__getitems__([index])
        return res[0]

    def _trace(self, msg: str):
        if self._trace_steps:
            print("[" + str(datetime.datetime.now()) + ", pid = " + str(os.getpid()) + "] " + msg)

    def __getitems__(self, indexes):
        self._trace("to __getitems__")
        if not self._augment:
            # If no augmentation delegate loading to ultralytics.data.dataset.YOLODataset.
            # Validation set request or train set without augmentation.
            res = []
            for index in indexes:
                result_label = copy.copy(super(DaliKorniaDataset, self).__getitem__(index))
                res.append(result_label)
            self._trace("from __getitems__")
            return res

        # Get file paths by indexes.
        labels = []
        labels_indexes = []
        can_apply_only_rect_safe_labels = []
        can_apply_only_rect_safe_labels_indexes = []
        for pos, i in enumerate(indexes):
            label = copy.copy(self.labels[i])
            if label['can_apply_only_rect_safe_transformations']:
                can_apply_only_rect_safe_labels.append(label)
                can_apply_only_rect_safe_labels_indexes.append(pos)
            else:
                labels.append(label)
                labels_indexes.append(pos)

        result = [None for i in range(len(indexes))]
        if len(labels) > 0:
            labels_result = self._get_items_by_labels(
                labels,
                apply_only_rect_safe_transformations=False
            )
            for labels_result_i, label_result in enumerate(labels_result):
                result[labels_indexes[labels_result_i]] = label_result

        if len(can_apply_only_rect_safe_labels) > 0:
            can_apply_only_rect_safe_labels_result = self._get_items_by_labels(
                can_apply_only_rect_safe_labels,
                apply_only_rect_safe_transformations=True
            )
            for labels_result_i, label_result in enumerate(can_apply_only_rect_safe_labels_result):
                result[can_apply_only_rect_safe_labels_indexes[labels_result_i]] = label_result

        assert len([x for x in result if x is None]) == 0

        self._trace("from __getitems__")
        return result

    def _evaluate_cut_segments_for_label(self, label):
        return nx.dataset.segment_utils.select_boxes_safe_for_cut(
            label['loaded_segments'],
            image_width=label['shape'][1],
            image_height=label['shape'][0],
            min_width=30,  # < don't apply object cut for very small objects.
            min_height=30,
        )

    def _get_cut_segments_for_label(self, label):
        key = label['im_file']
        if key not in self._cut_segments_for_labels:
            self._cut_segments_for_labels[key] = self._evaluate_cut_segments_for_label(label)
        return self._cut_segments_for_labels[key]

    def _is_masks_and_backgrounds_required(self):
        return (
            self._augment_args['replace_background_2x2'] > 0.00001 or
            self._augment_args['replace_background'] > 0.00001
        )

    def _get_items_by_labels(self, labels, apply_only_rect_safe_transformations=False):  # noqa: C901
        # Convert segments to keypoints for transform.
        all_keypoints = []
        files = []
        mask_files = []
        background_files = []
        if len(self.backgrounds) == 0:
            # Create simple background.
            black_image = np.zeros((self._imgsz, self._imgsz, 3), np.uint8)
            file_name = get_temp_image_name()
            cv2.imwrite(file_name, black_image)
            self.backgrounds.append({"im_file": file_name})

        self._trace("_get_items_by_labels: step 1")

        # Prepare images for object cut transformations.
        object_mosaic_probability = self._augment_args.get('object_mosaic', 0)
        object_overlay_mosaic_probability = self._augment_args.get('object_overlay_mosaic', 0)
        apply_object_mosaics = []
        cut_regions = []
        label_segments = []
        for label in labels:
            object_mosaic_mode = nx.dataset.mosaic_utils.ObjectMosaicMode.NONE
            if random.uniform(0, 1) < object_mosaic_probability:
                object_mosaic_mode = nx.dataset.mosaic_utils.ObjectMosaicMode.SIMPLE
            elif random.uniform(0, 1) < object_overlay_mosaic_probability:
                object_mosaic_mode = nx.dataset.mosaic_utils.ObjectMosaicMode.WITH_OVERLAY

            if object_mosaic_mode != nx.dataset.mosaic_utils.ObjectMosaicMode.NONE:
                # Evaluate or reuse regions(objects) that can be cutted from image.
                cut_boxes = self._get_cut_segments_for_label(label)
                if cut_boxes:
                    cut_box = random.choice(cut_boxes)
                    # adapt segments for 
                    cut_regions.append(cut_box)
                    label_segments.append(
                        nx.dataset.segment_utils.adapt_segments_for_cut(cut_box, label['loaded_segments'])
                    )
                else:
                    cut_regions.append((0., 0., 1., 1.))
                    label_segments.append(label['loaded_segments'])
                    object_mosaic_mode = nx.dataset.mosaic_utils.ObjectMosaicMode.NONE
            else:
                cut_regions.append((0., 0., 1., 1.))
                label_segments.append(label['loaded_segments'])

            apply_object_mosaics.append(object_mosaic_mode)

        self._trace("_get_items_by_labels: step 2")

        # Fill keypoints and mask files.
        for label, segments in zip(labels, label_segments):
            all_keypoints.append(self.polygon_segments_to_keypoints(segments))
            files.append(label['im_file'])
            if self._is_masks_and_backgrounds_required():
                mask_files.append(
                    label['mask_file'] if 'mask_file' in label and label['mask_file'] is not None
                    else self._full_mask_file
                )
                background_files.append(random.choice(self.backgrounds)['im_file'])

        torch.cuda.init()  # If we forked we need to initialize CUDA again.

        # select images for cut and mosaic object.
        # for these images we cut bbox inside dali pipeline (and do augmentation).
        self._trace("_get_items_by_labels: step 3")
        get_items_start_time = time.time()

        gc.disable()
        gc.collect()  # Run garbage collection manually for minimize maximum gpu memory usage at point of time.
        gc.enable()

        #self._trace("_get_items_by_labels: step 4")

        try:
          pipe = dali_load_and_augment_pipeline(
              files=files,
              mask_files=mask_files,
              background_files=background_files,
              empty_background=self._empty_background_file,
              keypoints=all_keypoints,
              cut_regions=cut_regions,
              batch_size=len(labels),
              num_threads=1,
              device_id=0,
              apply_only_rect_safe_transformations=apply_only_rect_safe_transformations,
              augment_args=self._augment_args,
              manual_garbage_collection=False, #True,
          )
          pipe.build()

          image_batch, label_batch, keypoints_batch = pipe.run()
          pipe = None  # < Free memory

          dali_tensor = image_batch.as_tensor()
          image_batch = None
          label_tensor = label_batch.as_tensor()
          label_batch = None

          images_torch_tensor = torch.empty(dali_tensor.shape(), dtype=torch.uint8, device=torch.device('cuda:0'))
          dali_plugin_pytorch.feed_ndarray(dali_tensor, images_torch_tensor)
          dali_tensor = None

          labels_torch_tensor = torch.empty(label_tensor.shape(), dtype=torch.int32, device=torch.device('cpu'))
          dali_plugin_pytorch.feed_ndarray(label_tensor, labels_torch_tensor)
          label_tensor = None

          keypoints = []
          for k in keypoints_batch:  # keypoints_batch is list of dali tensors.
              keyword_torch_tensor = torch.empty(k.shape(), dtype=torch.float32, device=torch.device('cpu'))
              dali.plugin.pytorch.feed_ndarray(k, keyword_torch_tensor)
              keypoints.append(keyword_torch_tensor.numpy().tolist())

          get_items_end_time = time.time()
          get_items_sum_time = get_items_end_time - get_items_start_time
          logger.debug("Getting items finished at " + str(get_items_sum_time) + "seconds")

          result = []
          corner_points = []
          for image_tensor, file_index_tensor, item_keypoints, segments in zip(
              images_torch_tensor,
              labels_torch_tensor,
              keypoints,
              label_segments  # < Decode keypoints with using segments used for fill it.
          ):
              image_shape = (image_tensor.shape[1], image_tensor.shape[2])  # < HW
              h, w = image_shape
              file_index = file_index_tensor.item()
              label = labels[file_index]  # < Here we use copy of label.

              # Fill segments by keypoints.
              segments, local_corner_points = self.keypoints_to_segments(
                  item_keypoints,
                  segments,
                  label=label,
              )
              corner_points.append(local_corner_points)

              """
              __getitems__ should return (h=640, w=481):
              im_file: str
              img: torch.tensor
              ori_shape: (h, w)
              resized_shape: (imgsz, imgsz)
              shape: (h, w)
              batch_idx: : torch.tensor([0.])
              cls: torch.tensor([[0.]])
              bboxes: torch.tensor([[0.6947, 0.5982, 0.4784, 0.7176]])
              """
              label['img'] = image_tensor
              label['ori_shape'] = image_shape
              label['shape'] = image_shape
              label['resized_shape'] = image_shape
              label['loaded_segments'] = segments
              label['ratio_pad'] = ((1.0, 1.0), (1, 1))

              result.append(label)

          self._trace("_get_items_by_labels: step 7")

          # Apply mosaic augmentation for images selected for object cut.
          for label, apply_object_mosaic, cut_region, corner_keypoints in zip(
                  result, apply_object_mosaics, cut_regions, corner_points
          ):
              if apply_object_mosaic != nx.dataset.mosaic_utils.ObjectMosaicMode.NONE:
                  # revert padding
                  _, height, width = label['img'].shape
                  x_left = min(c[0] for c in corner_keypoints)
                  x_right = max(c[0] for c in corner_keypoints)
                  y_top = min(c[1] for c in corner_keypoints)
                  y_bottom = max(c[1] for c in corner_keypoints)
                  x_left_int = int(x_left * width)
                  x_right_int = int(x_right * width)
                  y_top_int = int(y_top * height)
                  y_bottom_int = int(y_bottom * height)
                  segments = label['loaded_segments']
                  for s in segments:
                      if s.bbox is not None:
                          s.bbox[0] = (s.bbox[0] - x_left) / (x_right - x_left)
                          s.bbox[1] = (s.bbox[1] - y_top) / (y_bottom - y_top)
                          s.bbox[2] = (s.bbox[2] - x_left) / (x_right - x_left)
                          s.bbox[3] = (s.bbox[3] - y_top) / (y_bottom - y_top)
                      else:
                          s.polygon = [
                              ((p[0] - x_left) / (x_right - x_left), (p[1] - y_top) / (y_bottom - y_top))
                              for p in s.polygon
                          ]
                  label['img'] = label['img'][:, y_top_int:y_bottom_int, x_left_int:x_right_int]
                  label['img'], label['loaded_segments'] = nx.dataset.mosaic_utils.cut_and_paste_object_as_mosaic(
                      label['img'],
                      [(0, 0, 1, 1)],
                      segments,
                      result_width=640,
                      result_height=640,
                      mode=apply_object_mosaic
                  )

          self._trace("_get_items_by_labels: step 8")
          # Apply mixup augmentation (with using kornia)
          mixup3_probability = self._augment_args.get('mixup3', 0)
          mixup2_probability = self._augment_args.get('mixup2', 0)
          if len(result) > 1:
              for label_i, label in enumerate(result):
                  if random.uniform(0, 1) < mixup3_probability:
                      label_for_mix1 = DaliKorniaDataset.select_random_label(result, label_i)
                      label_for_mix2 = DaliKorniaDataset.select_random_label(result, label_i)
                      delta_coef = 0.02
                      mix_coef1 = random.uniform(0.33 - delta_coef, 0.33 + delta_coef)
                      mix_coef2 = random.uniform(0.33 - delta_coef, 0.33 + delta_coef)
                      label['img'] = (
                          label['img'].to(torch.float32) * mix_coef1 +
                          label_for_mix1['img'].to(torch.float32) * mix_coef2 +
                          label_for_mix2['img'].to(torch.float32) * (1 - mix_coef1 - mix_coef2)
                      ).to(torch.uint8)
                      label['loaded_segments'] = (
                          label['loaded_segments'] +
                          label_for_mix1['loaded_segments'] +
                          label_for_mix2['loaded_segments']
                      )
                  elif random.uniform(0, 1) < mixup2_probability:
                      label_for_mix = DaliKorniaDataset.select_random_label(result, label_i)
                      mix_coef = random.uniform(0.33, 0.66)
                      label['img'] = (
                          label['img'].to(torch.float32) * mix_coef +
                          label_for_mix['img'].to(torch.float32) * (1 - mix_coef)
                      ).to(torch.uint8)
                      label['loaded_segments'] = (label['loaded_segments'] + label_for_mix['loaded_segments'])

          # Fill bboxes and cls by result segments.
          for label in result:
              resized_shape = label['resized_shape']  # < HW
              h, w = resized_shape
              segments = copy.deepcopy(label['loaded_segments'])
              result_classes, result_bboxes, segments = nx.dataset.utils.segments_to_bboxes(
                  segments,
                  image_width=w,
                  image_height=h,
              )
              assert len(result_classes) == len(segments)
              label['cls'] = torch.tensor(result_classes)
              label['bboxes'] = torch.tensor(result_bboxes)
              # batch_idx should be batch index (index in requested items) repeated for each result object.
              label['batch_idx'] = torch.tensor([0] * len(result_classes), dtype=torch.float32)
              assert len(label['cls']) == len(segments)

          result_mem_usage = 0
          for label in result:
              img = label['img']
              result_mem_usage += img.element_size() * img.nelement()
          self._trace("_get_items_by_labels: step 10, gpu usage = " + str(result_mem_usage))

          return result

        finally:
          gc.disable()
          gc.collect()  # Run garbage collection manually for minimize maximum gpu memory usage at point of time.
          gc.enable()

    @staticmethod
    def select_random_label(labels, exclude_index):
        label_for_mix_i = random.randint(0, len(labels) - 2)
        label_for_mix_i = label_for_mix_i if label_for_mix_i < exclude_index else label_for_mix_i + 1
        return labels[label_for_mix_i]

    def polygon_segments_to_keypoints(self, segments) -> typing.List[typing.Tuple[float, float]]:
        keypoints = copy.deepcopy(self._corner_keypoints)
        for segment in segments:
            keypoints += [point for point in segment.polygon]
        return keypoints

    def keypoints_to_segments(
        self,
        keypoints: typing.List[typing.Tuple[float, float]],
        segments: typing.List[nx.dataset.utils.Segment],  # < original segments before trasform to keypoints.
        label = None,
    ) -> typing.Tuple[typing.List[nx.dataset.utils.Segment], typing.List[typing.Tuple[float, float]]]:
        # Convert keypoints to transformed segments:
        # segments is array of point arrays.
        # keypoints is numpy array of keypoints.
        # Transformation can increase the number of points multiple times.
        segments_points_count = 0
        for segment in segments:
            segments_points_count += len(segment.polygon)

        segment_portion_size = segments_points_count + len(self._corner_keypoints)
        assert len(keypoints) % segment_portion_size == 0, (
            "number of keypoints after trasformation " + str(len(keypoints)) +
            " is not a multiple of " + str(segment_portion_size) +
            "(segment points number + 4 control points)" +
            ((" on file: " + label["im_file"]) if label is not None else '')
        )

        keypoint_index = 0
        result_segments = []
        corner_points = []
        for kp_it in range(len(keypoints) // segment_portion_size):  # Process keypoints repeat blocks.
            corner_points += keypoints[
                keypoint_index: keypoint_index + len(self._corner_keypoints)
            ]
            keypoint_index += len(self._corner_keypoints)  # Skip corner points
            for segment_i, segment in enumerate(segments):
                result_segment = copy.deepcopy(segment)
                assert result_segment.polygon is not None   # < BaseDataset should provide only polygon segments.
                for point_index in range(len(result_segment.polygon)):
                    result_segment.polygon[point_index] = keypoints[keypoint_index]
                    keypoint_index += 1
                result_segments.append(result_segment)

        return result_segments, corner_points
