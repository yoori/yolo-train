"""
Create labelme annotations by predict result.
"""

import argparse
import logging
import cv2
import pathlib
import ultralytics

import nx.dataset.utils


logger = logging.getLogger(__name__)


def predict(image_file, min_conf = 0.1, max_size=None):
    image_file = pathlib.Path(image_file)
    image = cv2.imread(str(image_file))
    source_h, source_w, _ = image.shape
    if max_size is not None:
        image, _ = nx.dataset.utils.adapt_image_size(image, max_result_width=max_size, max_result_height=max_size)
    h, w, _ = image.shape
    predict_results = model.predict(image, conf=min_conf)
    segments = []
    for res in predict_results:
        for cls, conf, xyxy in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
            cls_index = round(cls.cpu().item())
            segments.append(nx.dataset.utils.Segment(
                label=(
                    str(cls_index) + ":" + str(int(conf * 100)) + "%"
                    if args.add_conf else str(cls_index)
                ),
                bbox=(xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h)
            ))
    result_labelme_file = str(pathlib.Path(image_file.parent) / pathlib.Path(image_file.stem + ".json"))
    nx.dataset.utils.write_labelme_annotations(
        result_labelme_file,
        segments=segments,
        image_path=image_file.name,
        image_width=source_w,
        image_height=source_h,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Default yolo11 detect script.')
    parser.add_argument('--image', help='image', type=str)
    parser.add_argument('--image-root', help='image root', type=str)
    parser.add_argument('--model', help='model', type=str, required=True)
    parser.add_argument('--conf', help='conf', type=float, default=0.5)
    parser.add_argument('--add-conf', help='Add confidence to label name', action='store_true')
    parser.add_argument('--size', help='Max image size (for resize before prediction)', type=int, default=640)
    parser.set_defaults(add_conf=False)
    args = parser.parse_args()

    model = ultralytics.YOLO(args.model)
    if args.image:
        predict(pathlib.Path(args.image), min_conf=args.conf, max_size=args.size)
    elif args.image_root:
        for image_file in pathlib.Path(args.image_root).rglob('*.jpg'):
            logger.info("Predict for " + str(image_file))
            predict(image_file, min_conf=args.conf, max_size=args.size)
    else:
        assert True, "One of --image, --image-root should be defined."
