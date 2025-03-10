import typing
import logging
import tempfile
import jinja2

import torch
import cv2
import ultralytics

import nx.dataset.base_dataset


logger = logging.getLogger(__name__)


def prepare_batch(validator, items):
    device = torch.device('cuda:0')
    batch = nx.dataset.base_dataset.collate_fn(items)
    # Run only actual steps from ultralytics.engine.validator.BaseValidator.__call__
    batch = validator.preprocess(batch)
    for k in ["img", "cls", "bboxes", "batch_idx"]:
        batch[k] = batch[k].to(device)
    return batch


def create_validator(
    validate_dir_root: str,
    device: str = 'cuda:0'):
    validator = ultralytics.models.yolo.detect.DetectionValidator(
        dataloader=None,
        args={
            'augment': False
        },
    )
    validator.device = torch.device(device)
    with tempfile.NamedTemporaryFile(suffix='.yaml') as data_file:
        with open(data_file.name, mode='w') as f:
            data_conf_str = jinja2.Template("""
path: {{datasetRoot}} # dataset root dir
train: train # train images (relative to 'path') 4 images
val: . # val images (relative to 'path') 4 images
test: # test images (optional)
scale: n

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
"""
            ).render({"datasetRoot": validate_dir_root})  # noqa: E124
            f.write(data_conf_str)
        validator.data = ultralytics.data.utils.check_det_dataset(data_file.name)

    validator.metrics.plot = False
    validator.args.plots = False
    return validator


def eval_mAP(validator, model, preds, batch):
    validator.init_metrics(model)
    preds = validator.postprocess(preds)
    validator.update_metrics(preds, batch)
    validator.get_stats()  # < Required for update metrics
    validator.finalize_metrics()
    local_mAP = validator.metrics.results_dict['metrics/mAP50-95(B)']
    return local_mAP


def get_labels_mAP(model, dataset, dataset_root, percent=0.1) -> typing.Dict[
    typing.Tuple[
        float,  # < mAP
        int  # < label index
    ],
    typing.Dict  # < label
]:
    device = torch.device('cuda:0')
    validator = create_validator(dataset_root)
    mAP_to_label = {}
    labels = dataset.get_labels()
    for label_index in range(len(labels)):
        # Run only actual steps from ultralytics.engine.validator.BaseValidator.__call__
        if label_index % 1000 == 0:
            logger.info("Processed " + str(label_index) + "/" + str(len(labels)) + " images")
        items = dataset.__getitems__([label_index])
        batch = prepare_batch(validator, items)
        preds = model(batch["img"], augment=False)
        detect_preds = preds[0]  # < Use only detection part.

        # Filter detections for only expected classes (for use util with standard yolo models).
        # preds : 0-3, 4: confidence, 5: class
        filtered_preds = []
        for p in detect_preds.squeeze().permute(1, 0).tolist():
            class_index = round(p[5])
            if class_index < 9:
                filtered_preds.append(p)
            else:
                filtered_preds.append([0] * len(p))
        detect_preds = torch.FloatTensor([filtered_preds])
        detect_preds = detect_preds.permute(0, 2, 1)
        detect_preds = detect_preds.to(device)

        local_mAP = eval_mAP(validator, model, [detect_preds], batch)
        mAP_to_label[(local_mAP, label_index)] = items[0]

    return mAP_to_label
