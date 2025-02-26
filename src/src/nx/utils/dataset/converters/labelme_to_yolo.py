import sys
import argparse
import pathlib
import logging

import nx.dataset.utils


logger = logging.getLogger(__name__)


def labelme_to_yolo(data_root):
    for json_file in sorted(pathlib.Path(data_root).resolve().rglob("*.json")):
        logger.info("Process file: " + str(json_file))
        if json_file.name != 'config.json':
            segments, image_width, image_height = nx.dataset.utils.read_labelme_annotations(json_file)
            result_yolo_file = str(json_file.parent / pathlib.Path(json_file.stem + ".txt"))
            logger.info("Save annotations to " + str(result_yolo_file))
            with open(result_yolo_file, 'w', encoding='utf-8') as f:
                for segment in segments:
                    min_x, min_y, max_x, max_y = segment.get_bbox()
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    width = max_x - min_x
                    height = max_y - min_y
                    f.write(
                        str(segment.label) + " " +
                        " ".join('{:,.3f}'.format(x) for x in [center_x, center_y, width, height]) + "\n"
                    )


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Util for convert labelme annotations to YOLO annotations.')
    parser.add_argument('--dataset-root', help = 'data root', required = True)
    args = parser.parse_args()
    labelme_to_yolo(args.dataset_root)
