"""
Script for filter and adapt google captcha set.

Google captcha set convert classes rule:
car -> 2 (car)
truck -> 7 (truck)
bus -> 5 (bus)
taxis -> 2 (car)
motocycle -> 3 (motorcycle)
bicycle -> 1 (bicycle)
boat -> 8 (boat)
tractors -> 2 (car)
bridge
chimneys
crosswalk
fire_hydrant
palm_trees
parking_meters
stair
traffic_light

and skip files where detected person (by default yolo detector).
"""

import sys
import argparse
import logging
import cv2
import pathlib
import ultralytics
import shutil

import nx.dataset.utils


logger = logging.getLogger(__name__)


LABEL_CONVERSIONS = {
    'car': 2,
    'truck': 7,
    'bus': 5,
    'taxis': 2,
    'motocycle': 3,
    'motorcycle': 3,
    'bicycle': 1,
    'boat': 8,
    'tractors': 2
}


for i in range(9):
    LABEL_CONVERSIONS[str(i)] = i


def filter_google_captcha_set(target_root, data_root, predict_model):
    for json_file in sorted(pathlib.Path(data_root).resolve().rglob("*.json")):
        image_file = json_file.parent / pathlib.Path(json_file.stem + ".jpg")
        image = cv2.imread(image_file)
        person_conf_threshold = 0.25
        predict_results = model.predict(image, conf=person_conf_threshold)
        person_present: bool = False
        for res in predict_results:
            for cls_tensor, conf, xyxy in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
                cls = round(cls_tensor.item())
                if (str(cls) == 'person' or str(cls) == '0') and conf > person_conf_threshold:
                    person_present = True

        if person_present:
            logger.info("Skip " + str(json_file) + ": person present")
            continue

        # Convert classes
        segments, image_width, image_height = nx.dataset.utils.read_labelme_annotations(
            json_file,
            string_labels=True
        )

        result_segments = []
        for segment in segments:
            if str(segment.label) in LABEL_CONVERSIONS:
                result_segments.append(segment)

        if len(result_segments) == 0:
            logger.info("Skip " + str(json_file) + ": no target classes present")
            continue

        shutil.copyfile(image_file, pathlib.Path(target_root) / (image_file.name))
        result_labelme_file = str(pathlib.Path(target_root) / pathlib.Path(image_file.stem + ".json"))
        nx.dataset.utils.write_labelme_annotations(
            result_labelme_file,
            image_path=image_file.name,
            image=image,
            segments=result_segments,
        )

        logger.info("Saved " + str(json_file))


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description='Default yolo11 detect script.')
    parser.add_argument('--model', help='model', type=str, required=True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )
    parser.add_argument('--data-root', help = 'Google captcha set root', required = True)
    args = parser.parse_args()

    model = ultralytics.YOLO(args.model)
    filter_google_captcha_set(args.target_root, args.data_root, model)
