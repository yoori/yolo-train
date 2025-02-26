"""
Utility for filter dataset.
this utility is used to highlight worst-case from dataset scenarios in which the current model
gives the greatest deviation. The resulting set can be added to the training set to improve the
quality of the model.
"""

import typing
import sys
import argparse
import pathlib
import logging
import shutil
import collections

import cv2
import ultralytics

import nx.dataset.utils


logger = logging.getLogger(__name__)


def group_images_by_min_confidence(
    root,
    min_predict_conf = 0.2,
    models = []
) -> typing.Dict[str, collections.OrderedDict[float, typing.List[pathlib.Path]]]:
    # < return class -> min confidence of this class on image
    result = {}

    file_i = 0
    for ann_file in pathlib.Path(root).glob('**/*.json'):
        if file_i > 0 and file_i % 1000 == 0:
            logger.info("Fetched " + str(file_i) + " files")
        file_i += 1
        image_file = ann_file.parent / (ann_file.stem + ".jpg")
        image = cv2.imread(str(image_file))
        assert image is not None, "Can't read annotated image: " + str(image_file)

        segments, _, _ = nx.dataset.utils.read_labelme_annotations(
            ann_file
        )

        min_conf_by_class = {}

        # Use min confidence by all models.
        for model in models:
            model_min_conf_by_class = {}
            for segment in segments:
                model_min_conf_by_class[segment.label] = None
            predict_results = model.predict(image, conf=min_predict_conf)
            for res in predict_results:
                for cls_tensor, conf_tensor in zip(res.boxes.cls, res.boxes.conf):
                    class_name = str(int(round(cls_tensor.item())))
                    conf = conf_tensor.item()
                    # print("Predicted " + class_name + ": " + str(conf))
                    if class_name not in model_min_conf_by_class:
                        # Excess object class detected - major case for this class.
                        model_min_conf_by_class[class_name] = 0
                    elif model_min_conf_by_class[class_name] is None:
                        model_min_conf_by_class[class_name] = conf
                    else:
                        model_min_conf_by_class[class_name] = min(model_min_conf_by_class[class_name], conf)
            for class_name, min_conf in model_min_conf_by_class.items():
                min_conf_by_class[class_name] = min(
                    min_conf_by_class[class_name] if class_name in min_conf_by_class else 1,
                    min_conf if min_conf is not None else 0
                )

        for class_name, min_conf in min_conf_by_class.items():
            if class_name not in result:
                result[class_name] = collections.OrderedDict()
            if min_conf not in result[class_name]:
                result[class_name][min_conf] = []
            result[class_name][min_conf].append(ann_file.parent / ann_file.stem)

    return result


def filter_images(target_root: typing.Union[str, pathlib.Path], root, class_shares = {}, models = None):
    target_root = pathlib.Path(target_root)
    logger.info("To group files by predicted confidence")
    images_by_class_and_min_conf = group_images_by_min_confidence(root, models=models)
    logger.info("To save files")
    copied_files = {}
    for class_name in sorted(images_by_class_and_min_conf.keys()):
        files_by_min_conf = images_by_class_and_min_conf[class_name]
        files_count = 0
        for _, file_pathes in files_by_min_conf.items():
            files_count += len(file_pathes)
        use_files_share = class_shares[class_name] if class_name in class_shares else 0
        to_use_files_count = int(files_count * use_files_share)
        copied_files_count = 0
        logger.info(
            "Use " + str(to_use_files_count) + "/" + str(files_count) + " files for class = " +
            str(class_name) + "(share = " + str(use_files_share) + ")"
        )
        copied_min_conf = 1
        copied_max_conf = 0
        for conf, file_pathes in files_by_min_conf.items():
            target_dir = target_root / str(class_name) / ("conf_" + str(min(int(conf * 100), 99)).zfill(2))
            target_dir.mkdir(parents=True, exist_ok=True)
            if copied_files_count >= to_use_files_count:
                break
            for file_path in file_pathes:
                if copied_files_count >= to_use_files_count:
                    break
                copied_min_conf = min(copied_min_conf, conf)
                copied_max_conf = max(copied_max_conf, conf)
                ann_file = file_path.parent / (file_path.stem + ".json")
                if ann_file not in copied_files:  # < File already copied for other class.
                    copied_files[ann_file] = True
                    shutil.copy(
                        ann_file,
                        str(target_dir / (file_path.stem + ".json"))
                    )
                    shutil.copy(
                        str(file_path.parent / (file_path.stem + ".jpg")),
                        str(target_dir / (file_path.stem + ".jpg"))
                    )
                    copied_files_count += 1
        logger.info(
            "For class = " +
            str(class_name) + " copied files with detection confidence in [" +
            str(int(copied_min_conf * 100) / 100) + ", " +
            str(int(copied_max_conf * 100) / 100) + "]"
        )


if __name__ == "__main__":
    ultralytics.utils.LOGGER.setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)

    logging.basicConfig(
        format='%(asctime)s [%(name)s] [%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(description='Util for filter annotated images.')
    parser.add_argument('--root', help = 'Open Images root', required = True)
    parser.add_argument(
        '-o', '--target-root',
        help = 'directory for save result',
        required = True
    )
    parser.add_argument('--model', nargs='+', help='model', type=str, required=True)
    parser.add_argument('--class-share', nargs='+', help='Share for class to use', action='append')

    args = parser.parse_args()

    class_shares = {}
    if args.class_share is not None:
        for class_share_str_arr in args.class_share:
            for class_share_str in class_share_str_arr:
                class_name, class_share = class_share_str.split(':')
                class_shares[class_name] = float(class_share)
    models = [ultralytics.YOLO(model) for model in args.model]
    filter_images(
        args.target_root,
        pathlib.Path(args.root),
        models=models,
        class_shares=class_shares,
    )
