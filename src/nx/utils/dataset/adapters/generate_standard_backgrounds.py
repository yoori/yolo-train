import sys
import argparse
import pathlib
import logging
import numpy as np
import cv2

import nx.dataset.utils


logger = logging.getLogger(__name__)


def create_blank(width, height, rgb_color=(0, 0, 0)):
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    color = tuple(reversed(rgb_color))
    image[:] = color
    return image


def generate_backgrounds(data_root, color = (255, 255, 255), color_name = 'white'):
    w = 640
    h = 640
    b = [0, 0, 0]
    c = list(reversed(list(color)))

    result_file = str(pathlib.Path(data_root) / ("background_" + color_name + '.jpg'))
    cv2.imwrite(result_file, create_blank(w, h, color))
    nx.dataset.utils.write_labelme_annotations(
        str(pathlib.Path(args.dataset_root) / ("background_" + color_name + '.json')),
        pathlib.Path("background_" + color_name + '.jpg'),
        image_width=w,
        image_height=h,
    )

    # chess 1x1
    result_file = str(pathlib.Path(data_root) / ("background_chess_" + color_name + '_1.jpg'))
    base = np.array([[b, c], [c, b]], dtype=np.uint8)
    chessboard_image = np.tile(base, (h // 2, w // 2, 1))
    cv2.imwrite(result_file, chessboard_image)
    nx.dataset.utils.write_labelme_annotations(
        str(pathlib.Path(args.dataset_root) / ("background_chess_" + color_name + '_1.json')),
        pathlib.Path("background_chess_" + color_name + '_1.jpg'),
        image_width=w,
        image_height=h,
    )

    # chess 2x2
    result_file = str(pathlib.Path(data_root) / ("background_chess_" + color_name + '_2.jpg'))
    chessboard_image = np.tile(np.array(
        [
            [b, b, c, c],
            [b, b, c, c],
            [c, c, b, b],
            [c, c, b, b],
        ], dtype=np.uint8
    ), (h // 4, w // 4, 1))
    cv2.imwrite(result_file, chessboard_image)
    nx.dataset.utils.write_labelme_annotations(
        str(pathlib.Path(args.dataset_root) / ("background_chess_" + color_name + '_2.json')),
        pathlib.Path("background_chess_" + color_name + '_2.jpg'),
        image_width=w,
        image_height=h,
    )

    # chess 4x4
    l1 = [b, b, b, b, c, c, c, c]
    l2 = [c, c, c, c, b, b, b, b]
    result_file = str(pathlib.Path(data_root) / ("background_chess_" + color_name + '_4.jpg'))
    chessboard_image = np.tile(np.array(
        [l1, l1, l1, l1, l2, l2, l2, l2], dtype=np.uint8
    ), (h // 8, w // 8, 1))
    cv2.imwrite(result_file, chessboard_image)
    nx.dataset.utils.write_labelme_annotations(
        str(pathlib.Path(args.dataset_root) / ("background_chess_" + color_name + '_4.json')),
        pathlib.Path("background_chess_" + color_name + '_4.jpg'),
        image_width=w,
        image_height=h,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Util for generate all objects mask by labelme annotations.')
    parser.add_argument('--dataset-root', help = 'data root', required = True)
    args = parser.parse_args()

    result_file = str(pathlib.Path(args.dataset_root) / "background_black.jpg")
    w = 640
    h = 640
    cv2.imwrite(result_file, create_blank(w, h, (0, 0, 0)))
    nx.dataset.utils.write_labelme_annotations(
        str(pathlib.Path(args.dataset_root) / "background_black.json"),
        pathlib.Path('background_black.jpg'),
        image_width=w,
        image_height=h,
    )

    generate_backgrounds(args.dataset_root, color=(255, 255, 255), color_name='white')
    generate_backgrounds(args.dataset_root, color=(255, 0, 0), color_name='red')
    generate_backgrounds(args.dataset_root, color=(0, 255, 0), color_name='green')
    generate_backgrounds(args.dataset_root, color=(0, 0, 255), color_name='blue')
    generate_backgrounds(args.dataset_root, color=(127, 127, 127), color_name='gray')
