"""
Utility for adapt dataset images sizes:
  * push to mosaic sizes less then target size and sizes that can be pushed to mosaic
    for target size (this allows you to reduce the load during training).
  * resize big size to target size with keeping ratio (this allows you to reduce the load
    on adapting pictures during training).
"""

import typing
import sys
import argparse
import pathlib
import logging
import shutil
import math
import numpy as np
import copy

import cv2

import nx.dataset.utils


logger = logging.getLogger(__name__)


def pad_image(image, segments, result_width = 40, result_height = 40):
    # Pad image and objects coordinates to center.
    h, w, _ = image.shape
    assert w <= result_width and h <= result_height
    if w == result_width and h == result_height:
        return image, segments

    padded_image = np.zeros((result_height, result_width, 3), np.uint8)
    padded_image[:] = (127, 127, 127)
    pos_x = int((result_width - w) / 2)
    pos_y = int((result_height - h) / 2)
    padded_image[pos_y:pos_y + h, pos_x:pos_x + w] = image
    image = padded_image

    result_segments = []
    for segment in segments:
        result_segment = copy.deepcopy(segment)
        if result_segment.bbox is not None:
            result_segment.bbox = (
                (result_segment.bbox[0] - 0.5) * w / result_width + 0.5,
                (result_segment.bbox[1] - 0.5) * h / result_height + 0.5,
                (result_segment.bbox[2] - 0.5) * w / result_width + 0.5,
                (result_segment.bbox[3] - 0.5) * h / result_height + 0.5,
            )
        else:
            result_segment.polygon = [
                [(p[0] - 0.5) * w / result_width + 0.5, (p[1] - 0.5) * h / result_height + 0.5]
                for p in result_segment.polygon
            ]
        result_segments.append(result_segment)

    return image, result_segments


def copy_non_same(src_path, dst_path):
    if src_path != dst_path:
        shutil.copy(src_path, dst_path)


def fill_mosaics(
    target_dir: typing.Union[str, pathlib.Path],
    source_dir: typing.Union[str, pathlib.Path],
    result_width: int = 640,
    result_height: int = 640,
    step_width: int = 40,
    step_height: int = 40,
    max_width_for_push_to_mosaic: int = None,  # < result_width / 2 by default
    max_height_for_push_to_mosaic: int = None,  # < result_height / 2 by default
    min_source_width: int = 25,
    min_source_height: int = 25,
):
    max_width_for_push_to_mosaic = (
        max_width_for_push_to_mosaic if max_width_for_push_to_mosaic is None
        else result_width / 2
    )
    max_height_for_push_to_mosaic = (
        max_height_for_push_to_mosaic if max_height_for_push_to_mosaic is None
        else result_height / 2
    )
    images_by_size = {}

    class_counts = {}

    # Group files by image size.
    for ann_file in pathlib.Path(source_dir).glob('**/*.json'):
        if ann_file.name == 'config.json':
            continue

        segments, image_width, image_height = nx.dataset.utils.read_labelme_annotations(ann_file)
        if image_width < min_source_width or image_height < min_source_height:
            logger.debug(
                "Skip very small image: " + str(ann_file.stem) + ": " + str(image_width) + "x" + str(image_height)
            )
            continue

        segments = [
            segment for segment in segments
            if nx.dataset.utils.filter_segment(segment, width=image_width, height=image_height)
        ]

        if len(segments) == 0:
            logger.debug("Skip by segments filtering")
            continue

        last_label = 'x'
        for segment in segments:
            if segment.label not in class_counts:
                class_counts[segment.label] = 0
            last_label = str(segment.label)
            class_counts[segment.label] += 1

        if (
            (image_height <= result_height / 2 and image_width <= result_width) and
            (image_width <= result_width / 2 and image_height <= result_height)
        ):
            # Push to mosaic.
            segments_only = True  # < Mosaic segments and bboxes separatly for allow rotate segments mosaic.
            for s in segments:
                if s.polygon is None:
                    segments_only = False
                    break
            round_w = int(math.ceil(image_width / step_width) * step_width)
            round_h = int(math.ceil(image_height / step_height) * step_height)
            key = (round_w, round_h, ("s" if segments_only else "r") + "_c" + str(last_label))
            if key not in images_by_size:
                images_by_size[key] = []
            images_by_size[key].append(ann_file.parent / ann_file.stem)
        else:
            # Adapt image size.
            image = cv2.imread(ann_file.parent / (ann_file.stem + ".jpg"))
            h, w, _ = image.shape
            assert w == image_width and h == image_height
            image, modified = nx.dataset.utils.adapt_image_size(
                image, max_result_width=result_width, max_result_height=result_height
            )
            if modified:
                new_h, new_w, _ = image.shape
                logger.debug(
                    "Adapt size for image: " + str(ann_file.stem) + ": " + str(w) + "x" + str(h) + " => " +
                    str(new_w) + "x" + str(new_h)
                )
                cv2.imwrite(pathlib.Path(target_dir) / (ann_file.stem + ".jpg"), image)
                # Resave annotations with new width, height.
                nx.dataset.utils.write_labelme_annotations(
                    pathlib.Path(target_dir) / ann_file.name,
                    segments=segments,
                    image_path=(ann_file.stem + ".jpg"),
                    image_width=new_w,
                    image_height=new_h,
                )
            else:
                #logger.debug("Save image as is: " + str(ann_file.stem))
                copy_non_same(
                    ann_file.parent / (ann_file.stem + ".jpg"),
                    pathlib.Path(target_dir) / str(ann_file.stem + ".jpg")
                )
                copy_non_same(
                    ann_file.parent / (ann_file.stem + ".json"),
                    pathlib.Path(target_dir) / str(ann_file.stem + ".json")
                )

    # Order mosaic candidates for get equal result between runs.
    for _, files in images_by_size.items():
        files.sort()

    # Create mosaic for each size.
    for size, files in images_by_size.items():
        while files:
            mosaic_image = np.zeros((result_height, result_width, 3), np.uint8)
            mosaic_image[:] = (127, 127, 127)
            adapted_segments = []
            used_files = []
            paste_x = 0
            paste_y = 0
            while files:
                paste_file = files.pop()
                if result_width - paste_x < size[0]:
                    paste_x = 0
                    paste_y += size[1]
                if result_height - paste_y < size[1]:
                    files.append(paste_file)
                    break  # < Create new image, current filled.
                image = cv2.imread(paste_file.parent / str(paste_file.name + ".jpg"))
                assert image is not None
                loaded_h, loaded_w, _ = image.shape
                segments, ann_w, ann_h = nx.dataset.utils.read_labelme_annotations(
                    paste_file.parent / (paste_file.name + ".json")
                )
                assert loaded_w == ann_w and loaded_h == ann_h

                # Pad image for mosaic cell.
                image, segments = pad_image(image, segments, result_width=size[0], result_height=size[1])
                h, w, _ = image.shape
                assert w == size[0] and h == size[1]

                # Adapt coordinates for paste.
                for s in segments:
                    if s.bbox is not None:
                        s.bbox = (
                            (paste_x + s.bbox[0] * size[0]) / result_width,
                            (paste_y + s.bbox[1] * size[1]) / result_height,
                            (paste_x + s.bbox[2] * size[0]) / result_width,
                            (paste_y + s.bbox[3] * size[1]) / result_height,
                        )
                    else:
                        s.polygon = [
                            [(paste_x + p[0] * size[0]) / result_width, (paste_y + p[1] * size[1]) / result_height]
                            for p in s.polygon
                        ]
                adapted_segments += segments
                mosaic_image[paste_y:paste_y + size[1], paste_x:paste_x + size[0]] = image
                used_files.append(paste_file)
                paste_x += size[0]

            assert len(used_files) > 0
            # used_files contains file paths in format : <source folder>/<file stem>

            if len(used_files) > 1:
                result_file = (
                    "mosaic" + str(int(result_width / size[0])) + "x" +
                    str(int(result_height / size[1])) +
                    "_" + size[2] +
                    "_" + "_".join([str(f.stem) for f in used_files])
                )[0:100]  # < File name too long workaround.
                logger.debug("Create mosaic: " + str(result_file))
                cv2.imwrite(pathlib.Path(target_dir) / (result_file + ".jpg"), mosaic_image)
                nx.dataset.utils.write_labelme_annotations(
                    pathlib.Path(target_dir) / (result_file + ".json"),
                    segments=adapted_segments,
                    image_path=(result_file + ".jpg"),
                    image_width=result_width,
                    image_height=result_height,
                )
            else:  # < No other images for push to masaic, store image as is.
                result_file = used_files[0].name
                logger.debug("Save mosaic candidate as is: " + str(result_file))
                copy_non_same(
                    used_files[0].parent / (used_files[0].name + ".jpg"),
                    pathlib.Path(target_dir) / str(result_file + ".jpg"),
                )
                copy_non_same(
                    used_files[0].parent / (used_files[0].name + ".json"),
                    pathlib.Path(target_dir) / str(result_file + ".json"),
                )

    logger.info(
        "Saved instances:\n  " + str("\n  ").join([
            (class_name.rjust(10) + ": " + str(class_counts[class_name]))
            for class_name in sorted(class_counts.keys())
        ])
    )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(
        description="""Util for pack small images to mosaics and resize big images to target size.""")
    parser.add_argument('--root', help = 'dataset root', required = True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )
    args = parser.parse_args()

    fill_mosaics(args.target_root, args.root)
