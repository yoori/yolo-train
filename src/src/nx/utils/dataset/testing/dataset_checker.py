
"""
Util for check dataset wrapper and augmentations.
"""
import torch
import cv2
import argparse
import time
import json
import pathlib

import nx.dataset.utils
from nx.dataset.albumentation_dataset import AlbumentationDataset
from nx.dataset.dali_kornia_dataset import DaliKorniaDataset


class ConfigStub(object):
    mosaic = 0.0
    mixup = 0.0
    bgr = False
    mask_ratio = None
    overlap_mask = None
    copy_paste = 0.0
    degrees = 0
    # augment=True requirements:
    translate = 0.1
    scale = 0.5
    shear = 0.0
    perspective = 0.0
    copy_paste_mode = 'flip'
    hsv_h = 0.5
    hsv_s = 0.5
    hsv_v = 0.5
    flipud = 0.1  # flip probability.
    fliplr = 0.1  # h flip probability.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Default yolo11 model training script.')
    parser.add_argument(
        '-d', '--dataset-root',
        help = 'data set root',
        required=True
    )
    parser.add_argument('--output', help='output folder', type=str, required=True)
    parser.add_argument('--ops', help='number of calls to dataset', type=int, default=1)
    parser.add_argument('--batch', help='batch size', type=int, default=1)
    parser.add_argument('--mode', help='mode', type=str, choices=['albumentation', 'dali'], default='albumentation')
    parser.add_argument('--augment-config', help='augmentation config', type=str, default=None)

    parser.set_defaults()
    args = parser.parse_args()

    augment_args = None
    if args.augment_config is not None:
        file_content = pathlib.Path(args.augment_config).read_text()
        augment_args = json.loads(file_content)

    dataset_args = {
        'img_path': args.dataset_root,
        'imgsz': 640,
        'batch_size': 32,
        'augment': True,
        'hyp': ConfigStub(),
        'rect': False,
        'cache': None,
        'single_cls': False,
        'stride': 32,
        'pad': 0.0,
        'prefix': "",
        'classes': None,
        'data': {
            'path': args.dataset_root,
            'train': args.dataset_root,
            'val': args.dataset_root,
            'nc': 9
        },
        'fraction': 1.0,
    }

    if augment_args is not None:
        dataset_args['augment_args'] = augment_args

    if args.mode == 'albumentation':
        dataset = AlbumentationDataset(**dataset_args)
    else:
        dataset = DaliKorniaDataset(**dataset_args)

    all_labels = dataset.get_labels()
    max_index = len(all_labels)

    # Fetch dataset and resave it to output folder.
    get_items_sum_time = 0
    op_index = 0
    for i in range(args.ops):
        indexes = [(i * args.batch + batch_i) % max_index for batch_i in range(args.batch)]
        get_items_start_time = time.time()
        items = dataset.__getitems__(indexes)
        get_items_end_time = time.time()
        get_items_sum_time = get_items_end_time - get_items_start_time
        for item, label_index in zip(items, indexes):
            label = all_labels[label_index]
            image_tensor: torch.tensor = item['img']  # < Image after transormation to tensor.
            #print("image_tensor.shape = " + str(image_tensor.shape))
            #print("image_tensor.dtype = " + str(image_tensor.dtype))
            #print("image_tensor.device = " + str(image_tensor.device))
            image = image_tensor.detach().cpu().numpy()
            image = image.transpose(1, 2, 0)
            image = image[:, :, ::-1]  # < BGR -> RGB, ultralytics dataset's return image in BGR.
            loaded_classes = item['cls']
            # loaded_segments can absent after some specific transformations (mosaic, ...).
            loaded_segments = item['loaded_segments'] if 'loaded_segments' in item else []

            yolo_bboxes = item['bboxes'].numpy().tolist()
            res_file = args.output + "/" + str(op_index) + ".jpg"
            print("to save image for '" + str(label['im_file']) + "' into '" + res_file + "'")
            save_success = cv2.imwrite(res_file, image)
            assert save_success

            save_bboxes = []
            for yolo_bbox, bb_label in zip(yolo_bboxes, loaded_classes):
                center_x, center_y, bb_width, bb_height = yolo_bbox
                save_bboxes.append(
                    nx.dataset.utils.Segment(
                        label=str(int(bb_label[0])) + "_bbox",
                        bbox=[
                            center_x - bb_width / 2,
                            center_y - bb_height / 2,
                            center_x + bb_width / 2,
                            center_y + bb_height / 2,
                        ]
                    )
                )

            nx.dataset.utils.write_labelme_annotations(
                args.output + "/" + str(op_index) + ".json",
                image_path = str(op_index) + ".jpg",
                segments = save_bboxes,
                image = image
            )
            op_index += 1

    print("Get items sum time: " + str(get_items_sum_time))
