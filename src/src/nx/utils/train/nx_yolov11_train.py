"""
NX model training utility:
  * much augmentations with using nvidia.dali (faster then standard augmentations at 15-20 times).
  * accurate bbox position after geomethric conversions (we use segmentation inside if it is available).
  * use extended dataset.
  * train with random hyperparameters at train attempts.

Now we use on host with GPU memory = 24Gb:
For train:
python3.9 nx_yolov11_train.py --dataset-root datasets/coco/ --batch 30 --model <trained nx model> --epochs 50 \
  --resume -mode dali --train-attempts 10 --data-loader-workers 6

For resume traning:
python3.9 nx_yolov11_train.py --dataset-root datasets/coco/ --batch 30 --epochs 50 \
  --resume-model=<last.pt> -mode dali --train-attempts 10 --data-loader-workers 6

For train model from zero use adapted original yolo model as base:
# Adapt original yolo model to class number change:
yolo train data=./nx.yaml model=./model.yaml pretrained=models/yolo11n_orig.pt epochs=10 \
  lr0=0.01 batch=200 box=18.27875 cls=1.32899 dfl=0.56016 freeze=22 workers=16

yolo train data=./nx.yaml model=./model.yaml pretrained=models/yolo11n_orig.pt epochs=10 \
  lr0=0.01 batch=200 box=18.27875 cls=1.32899 dfl=0.56016 freeze=21 workers=16

"""

import copy
import random
import tempfile
import shutil
import jinja2
import logging
import multiprocessing
import argparse
import json
import pathlib
import enum
import dataclasses
import torch
import torch.quantization
import ultralytics
import ultralytics.data.build

import nx.dataset.base_dataset
from nx.dataset.albumentation_dataset import AlbumentationDataset
from nx.dataset.dali_kornia_dataset import DaliKorniaDataset


logger = logging.getLogger(__name__)


NX_YOLO_MODEL_CONFIG = """
# Parameters
nc: 9 # number of classes

scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs

#  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
#  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
#  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
#  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]  # level #20
  - [[-1, 10], 1, Concat, [1]] # cat head P5  # level #21
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)  # level #22

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)  # level #23
"""


class NxCustomDataLoader(ultralytics.data.build.InfiniteDataLoader):
    """
    class Iterator:
        def __init__(self, it):
            self.it = it

        def __iter__(self):
            return self

        def __next__(self):
            return self.it.__next__()
    """

    def __init__(
        self, dataset, batch_size=8, shuffle=True, transforms=None, rank=None, mode=None,
        num_workers: int = 12
    ):
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=None,
            collate_fn=nx.dataset.base_dataset.collate_fn,
            generator=generator,
        )

class NxCustomTrainer(ultralytics.models.yolo.detect.train.DetectionTrainer):
    @dataclasses.dataclass
    class EpochMode:
        class Mode(enum.Enum):
            ALL=1
            WORST_MAP=2
        mode: 'NxCustomTrainer.EpochMode.Mode' = 1
        percent: float = 1.0


    def __init__(
        self, *args,
        dataset_mode = 'albumentation',
        data_loader_workers = 12,
        augment = True,
        augment_args = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._dataset_mode = dataset_mode
        self._data_loader_workers = data_loader_workers
        self._augment = augment
        self._augment_args = augment_args
        self.add_callback('on_train_epoch_start', lambda x: self.on_start_epoch_())
        self.epoch_plan = [
            NxCustomTrainer.EpochMode(
                mode=NxCustomTrainer.EpochMode.Mode.ALL,
                percent=1.0,
            )
        ] * 1000

    def on_start_epoch_(self):
        epoch = self.epoch
        # set dataset mode, one of: full, worst_map(percent)

    def plot_training_labels(self):
        return super().plot_training_labels()

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        cfg = self.args
        dataset_args = {
            'img_path': dataset_path,
            'imgsz': cfg.imgsz,
            'batch_size': batch_size,
            'augment': (mode == "train" and self._augment),
            'hyp': cfg,
            'rect': cfg.rect,
            'cache': cfg.cache or None,
            'single_cls': cfg.single_cls or False,
            'stride': 32,
            'prefix': f"{mode}: ",
            'classes': cfg.classes,
            'data': self.data,
            'fraction': cfg.fraction if mode == "train" else 1.0,
        }
        if self._dataset_mode == 'dali':
            dataset = DaliKorniaDataset(
                **dataset_args,
                augment_args=self._augment_args,
            )
        elif self._dataset_mode == 'albumentation':
            dataset = AlbumentationDataset(**dataset_args)
        else:
            dataset = ultralytics.YOLODataset(**dataset_args)
        self.dataset = dataset
        return NxCustomDataLoader(
            dataset,
            batch_size=batch_size,
            rank=rank,
            mode=mode,
            num_workers=self._data_loader_workers,
        )

    def train_dataloader(self, *args, **kwargs):
        return self.get_dataloader(*args, **kwargs, is_train=True)

    def val_dataloader(self, *args, **kwargs):
        return self.get_dataloader(*args, **kwargs, is_train=False)


def val(model_file, weights_file, data_file, args=None):
    val_model = ultralytics.YOLO(model_file)
    val_model = val_model.load(weights_file)
    dataset_args = {
        'img_path': args.dataset_root + "/val",
        'imgsz': 640,
        'augment': False,
    }
    # We expect equal metrics on validation set for all dataset implementations.
    if args.mode == 'dali':
        dataset = DaliKorniaDataset(**dataset_args)
    elif args.mode == 'albumentation':
        dataset = AlbumentationDataset(**dataset_args)
    else:
        dataset = ultralytics.YOLODataset(**dataset_args)
    dataloader = NxCustomDataLoader(
        dataset,
        batch_size=8,
        num_workers=1,
    )

    print("XXX args: " + str(args))
    metrics = val_model.val(
        data=data_file,
        validator=lambda args, _callbacks: ultralytics.models.yolo.detect.DetectionValidator(
            dataloader=dataloader,
            args=args
        ),
        mode='val'
    )
    return metrics.results_dict['metrics/mAP50-95(B)']


def prepare_args_parser():
    parser = argparse.ArgumentParser(description='Default yolo11 model training script.')
    subparsers = parser.add_subparsers(dest='command', help='subcommand help')

    # Configure 'train' command.
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--epochs', help='number of epochs', type=int, default=100)
    train_parser.add_argument('--batch', help='batch size', type=int, default=64)
    train_parser.add_argument('--data-loader-workers', help='number of data loading workers', type=int, default=1)
    train_parser.add_argument('--model', help='load model', type=str, default=None)
    train_parser.add_argument(
        '--no-augment', help='disable augmentations', dest='augment',
        action='store_false'
    )
    train_parser.add_argument('-d', '--dataset-root', help = 'data set root', required=True)
    train_parser.add_argument(
        '--mode', help='mode', type=str,
        choices=[
            'albumentation',
            'dali',
            'yolo'
        ],
        default='dali'
    )
    train_parser.add_argument('--train-attempts', help='train attempts', type=int, default=1)
    train_parser.add_argument('--save-best', help='path to save best model', type=str, default='best.pt')
    train_parser.add_argument('--lr0', help='lr0', type=float, default=None)
    train_parser.add_argument('--lrf', help='lrf', type=float, default=None)
    train_parser.add_argument('--freeze', help='freeze', type=int, default=None)
    train_parser.add_argument('--augment-config', help='augmentation config', type=str, default=None)
    train_parser.add_argument(
        '--resume-model', help='resume training by state in this model',
        type=str, default=None,
    )

    train_parser.set_defaults(augment=True, resume=False)

    # Configure 'validate' command.
    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument('--model', help='load model', type=str, required=True)
    validate_parser.add_argument(
        '-d', '--dataset-root',
        help = 'data set root',
        required=True
    )
    validate_parser.add_argument(
        '--mode', help='mode', type=str,
        choices=[
            'albumentation',
            'dali',
            'yolo'
        ],
        default='dali'
    )

    # Configure 'tune' command.
    tune_parser = subparsers.add_parser('tune')
    tune_parser.add_argument('--epochs', help='number of epochs', type=int, default=100)

    return parser


def main():  # noqa: C901
    multiprocessing.set_start_method('spawn')  # Use spawn for use CUDA inside forked data loaders.

    parser = prepare_args_parser()
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix='.yaml') as modelFile, \
        tempfile.NamedTemporaryFile(suffix='.yaml') as dataFile:  # noqa: E125
        logger.info("Model file: " + str(modelFile.name))
        with open(modelFile.name, mode='w') as f:
            f.write(NX_YOLO_MODEL_CONFIG)

        with open(dataFile.name, mode='w') as f:
            dataConfStr = jinja2.Template("""
path: {{datasetRoot}} # dataset root dir
train: train # train images (relative to 'path') 4 images
val: val # val images (relative to 'path') 4 images
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
            ).render({"datasetRoot": args.dataset_root})  # noqa: E124
            f.write(dataConfStr)

        print("XXX0 args = " + str(args))
        if args.command == 'train':
            add_args = {}
            if args.resume:
                add_args['resume'] = True
            if args.freeze is not None:
                add_args['freeze'] = args.freeze
            best_model_file = args.model

            for attempt_i in range(args.train_attempts):
                if args.resume_model is None:
                    if args.model:
                        logger.warning(
                            """For resume training model argument will be skipped,
                            state of model should be loaded from --resume-model)"""
                        )
                    # Load model structure description.
                    model = ultralytics.YOLO(modelFile.name)
                    if best_model_file:
                        model.load(best_model_file)
                        add_args['pretrained'] = best_model_file
                else:
                    # Resume training, get structure and training state(optimizer traits, ...) from passed model.
                    # Without resume we only read passed model(weights) as start state of model for training.
                    model = ultralytics.YOLO(args.resume_model)
                    add_args['model'] = args.resume_model
                    add_args['resume'] = True

                local_add_args = copy.copy(add_args)
                if True:  # attempt_i == 0:
                    local_add_args.update({
                        'lr0': (args.lr0 if args.lr0 is not None else 0.00269),
                        'lrf': (args.lrf if args.lrf is not None else 0.00288),
                        'momentum': 0.73375,
                        'weight_decay': 0.00015,
                        'warmup_epochs': 1.22935,
                        'warmup_momentum': 0.1525,
                        'box': 18.27875,
                        'cls': 1.32899,
                        'dfl': 0.56016,
                    })
                else:
                    local_add_args.update({
                        'lr0': (args.lr0 if args.lr0 is not None else random.uniform(0.002152, 0.003228)),
                        'lrf': (args.lrf if args.lrf is not None else random.uniform(0.002304, 0.003456)),
                        'momentum': random.uniform(0.660375, 0.807125),
                        'weight_decay': random.uniform(0.00005, 0.00025),
                        'warmup_epochs': random.uniform(0.983480, 1.475220),
                        'warmup_momentum': random.uniform(0.12200, 0.18300),
                        'box': random.uniform(15.550875, 19.006625),
                        'cls': random.uniform(1.196091, 1.461889),
                        'dfl': random.uniform(0.504144, 0.616176),
                    })
                logger.info(
                    "Train step with arguments: " +
                    ", ".join([str(k) + " = " + str(v) for k, v in local_add_args.items()])
                )
                local_add_args.update({
                    # Hyperparameters from https://docs.ultralytics.com/guides/hyperparameter-tuning/#best_hyperparametersyaml
                    'optimizer': "AdamW",
                    # Next augmentations traits disabled in dali mode.
                    'hsv_h': 0.01148,
                    'hsv_s': 0.53554,
                    'hsv_v': 0.13636,
                    'degrees': 0.0,
                    'translate': 0.12431,
                    'scale': 0.07643,
                    'shear': 0.0,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.08631,
                    'mixup': 0.0,
                    'copy_paste': 0.0,
                })

                start_mAP = 0
                if best_model_file:
                    start_mAP = val(modelFile.name, best_model_file, dataFile.name, args=args)
                    pass
                logger.info(
                    "Start training attempt #" + str(attempt_i) + ": metrics/mAP50-95(B) = " +
                    str(start_mAP)
                )

                augment_args = None
                if args.augment_config is not None:
                    file_content = pathlib.Path(args.augment_config).read_text()
                    augment_args = json.loads(file_content)

                # Add QAT only if it isn't attached already.
                model_qconfig = getattr(model, "qconfig", None)
                if model_qconfig is None:
                    model.training = True
                    torch.quantization.prepare_qat(model, inplace=True)

                results = model.train(
                    data=dataFile.name,
                    epochs=args.epochs,
                    batch=args.batch,
                    imgsz=640,
                    name="train_nx_yolov11n",
                    workers=12,
                    patience=20,
                    device='cuda:0',
                    augment=args.augment,
                    trainer=lambda overrides, _callbacks: NxCustomTrainer(
                        dataset_mode=args.mode,
                        data_loader_workers=args.data_loader_workers,
                        overrides=overrides,
                        augment=args.augment,
                        augment_args=augment_args,
                    ),
                    **local_add_args
                )

                # Check metrics if it is better then start state - use it.
                result_mAP = results.results_dict['metrics/mAP50-95(B)']
                if result_mAP < start_mAP and best_model_file:
                    msg = 'improve failed'
                else:
                    model.save(args.save_best)
                    best_model_file = args.save_best
                    msg = 'improved'
                logger.info(
                    "Finish training attempt #" + str(attempt_i) + ", model " + msg + ": metrics/mAP50-95(B) = " +
                    str(result_mAP)
                )
                try:
                    shutil.move('runs/detect', "runs/attempt_" + str(attempt_i))
                except Exception:
                    pass
        elif args.command == 'validate':
            mAP = val(modelFile.name, args.model, dataFile.name, args=args)
            print("metrics/mAP50-95(B): " + str(mAP))
        elif args.command == 'tune':
            # To recheck
            model = ultralytics.YOLO(best_model_file)
            model.tune(
                data=modelFile.name,
                epochs=args.epochs,
                iterations=20,
                optimizer="AdamW",
                plots=False,
                save=False,
                val=False
            )


if __name__ == "__main__":
    main()
