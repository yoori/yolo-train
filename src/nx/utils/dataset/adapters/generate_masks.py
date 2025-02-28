import sys
import argparse
import pathlib
import logging
import numpy as np
import cv2

import nx.dataset.utils


logger = logging.getLogger(__name__)


def generate_masks(data_root):
    for json_file in sorted(pathlib.Path(data_root).resolve().rglob("*.json")):
        print("Process file: " + str(json_file))
        logger.info("Process file: " + str(json_file))
        if json_file.name != 'config.json':
            segments, image_width, image_height = nx.dataset.utils.read_labelme_annotations(json_file)

            mask_image = np.zeros((image_height, image_width, 3), dtype = np.uint8)
            for segment in segments:
                segment = segment.convert_to_polygon_segment()
                cv2.fillPoly(
                    mask_image,
                    pts=[np.array([[int(p[0] * image_width), int(p[1] * image_height)] for p in segment.polygon])],
                    color=(255, 255, 255)
                )

            result_mask_file = str(json_file.parent / pathlib.Path(json_file.stem + "_mask.png"))
            cv2.imwrite(result_mask_file, mask_image)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Util for generate all objects mask by labelme annotations.')
    parser.add_argument('--dataset-root', help = 'data root', required = True)
    args = parser.parse_args()
    generate_masks(args.dataset_root)
