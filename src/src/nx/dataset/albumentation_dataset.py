import numpy as np
import logging
import cv2
import torch
import copy

import ultralytics
import ultralytics.data.dataset
import ultralytics.utils.instance
import albumentations as A

import nx.dataset.utils
import nx.dataset.base_dataset


logger = logging.getLogger(__name__)


class RectUnsafeTransformationsWrapper(object):
    """
    Wrapper for rect non safe trasformations,
    that apply trasformations only for images where we have segmentation annotations.
    As example, rect non safe trasformations: rotations, perspective, fisheye
    """
    albumentation = None

    def __init__(self, albumentation):
        self._albumentation = albumentation

    def __call__(self, data):
        if 'can_apply_only_rect_safe_transformations' not in data or not data['can_apply_only_rect_safe_transformations']:
            return self._albumentation(data)
        else:
            return data  # < Don't apply rect non safe transformations.


class AlbumentationWrapper(object):
    """
    Wrapper for albumentation transformer.
    """
    albumentation = None
    _name: str = None

    def __init__(self, albumentation, name = None):
        self._albumentation = A.Compose(
            albumentation,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
            bbox_params=A.BboxParams(format='yolo', label_fields=['cls'])
        )
        self._name = str(albumentation) if name is None else name

    def __call__(self, data):
        try:
            result_data = copy.copy(data)
            # Transform loaded_segments to keypoints and back after transformation.
            image = result_data['img']
            image_height, image_width, _ = image.shape
            loaded_segments = copy.deepcopy(result_data["loaded_segments"])
            keypoints = []
            for segment in loaded_segments:
                assert segment.polygon is not None  # < BaseDataset should transform all segments to polygons.
                for point in segment.polygon:
                    keypoints.append([
                        min(round(point[0] * image_width), image_width - 1),
                        min(round(point[1] * image_height), image_height - 1)
                    ])
            # Fill bboxes by transformed segments,
            # YOLO dataset keep it in instances (see YOLODataset.update_labels_info - it called before transforms).
            result_data['cls'], bboxes, loaded_segments = nx.dataset.utils.segments_to_bboxes(
                loaded_segments,
                image_width=image_width,
                image_height=image_height,
            )
            transformResult = self._albumentation(
                image=image,
                bboxes=bboxes,
                keypoints=keypoints,
                cls=result_data['cls']
            )
            resultImage = transformResult['image']
            result_image_height, result_image_width, _ = resultImage.shape
            keypoints = transformResult['keypoints']

            # Reconstruct loaded_segments by keypoints (after geometric transformations).
            keypoint_i = 0
            for segment in loaded_segments:
                for point_index, _ in enumerate(segment.polygon):
                    segment[point_index] = [
                        min(max(float(keypoints[keypoint_i][0]) / result_image_width, 0.000001), 0.999999),
                        min(max(float(keypoints[keypoint_i][1]) / result_image_height, 0.000001), 0.999999)
                    ]
                    keypoint_i += 1

            result_data['shape'] = (result_image_height, result_image_width)
            result_data['img'] = resultImage
            result_data['loaded_segments'] = loaded_segments
            # Fill more accurate bboxes by loaded_segments.
            result_data['cls'], bboxes, _ = nx.dataset.utils.segments_to_bboxes(
                result_data["loaded_segments"],
                image_width=result_image_width,
                image_height=result_image_height,
            )

            assert "instances" in result_data
            instances = result_data['instances']
            result_data['instances'] = ultralytics.utils.instance.Instances(
                bboxes=bboxes,
                segments=instances.segments,
                keypoints=instances.keypoints,
                normalized=True
            )

            return result_data
        except Exception as e:
            raise Exception("Error on apply " + self._name + ": " + str(e)) from e


class ClosingAugmentation(object):
    """
    Augmentation for inject before builtin augmentations.
    * Allow to transform segments by bounds changes in builtin augmentations.
    * Fill bboxes by segments.
    """
    def __init__(self, add_control_bbox: bool = False):
        self._add_control_bbox = add_control_bbox

    def __call__(self, data):
        result_data = dict(data)
        loaded_segments = result_data["loaded_segments"]
        classes, bboxes, _ = nx.dataset.utils.segments_to_bboxes(
            result_data['cls'], loaded_segments
        )

        assert len(classes) == len(bboxes)

        if self._add_control_bbox:
            # add control bbox - it should be removed at __getitem__ or __getitems__
            classes = np.append(classes, np.array([[-1]], dtype=np.float32), axis=0)
            bboxes = np.append(bboxes, np.array([[0.5, 0.5, 1, 1]], dtype=np.float32), axis=0)

        result_data['cls'] = classes
        original_instances = data.get('instances', None)
        result_data['instances'] = ultralytics.utils.instance.Instances(
            bboxes=bboxes,
            segments=(original_instances.segments if original_instances else None),
            keypoints=(original_instances.keypoints if original_instances else None),
            normalized=True
        )

        return result_data


class AlbumentationDataset(nx.dataset.base_dataset.BaseDataset):
    """
    * Override get_labels for load labelme markup files.
    * Customize augmentations by override build_transforms: it should convert segments to bboxes.
    """

    def __init__(self, *args, **kwargs):
        self._add_control_bbox = False
        super().__init__(*args, **kwargs)

    def get_labels(self):
        labels = super().get_labels()
        for label in labels:
            label['normalized'] = True
            label['bbox_format'] = 'xywh'
        return labels

    def build_transforms(self, hyp=None):
        """
        NX albumentations:
          NxAugmentations
          SegmentsToBbox: convert segments to bboxes after NxAugmentations
        Default YOLO albumentations:
          Blur(p=0.01, blur_limit=(3, 7)),
          MedianBlur(p=0.01, blur_limit=(3, 7)),
          ToGray(p=0.01, num_output_channels=3, method='weighted_average'),
          CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
        """
        transforms = super().build_transforms(hyp=hyp)
        add_augmentations = []

        if self.augment:
            # Insert custom transformations before standard ! (standard transform image to tensor)
            # Geometric transformations => BBox based transformations => Color and quality transformations
            add_augmentations.append(AlbumentationWrapper(
                A.Compose(
                    [
                        # channel manipulations.
                        A.ChannelDropout(channel_drop_range=(1, 1), fill=128, p=0.1),
                        A.ChannelShuffle(p=0.1),
                        #
                        A.ToGray(num_output_channels=3, method='weighted_average', p=0.05),

                        A.Blur(blur_limit=(3, 7), p=0.05),
                        A.MedianBlur(blur_limit=(3, 7), p=0.05),
                        A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.05),

                        A.RandomRain(
                            slant_lower=0,
                            slant_upper=0,
                            brightness_coefficient=1.0,
                            drop_length=2,
                            drop_width=2,
                            drop_color=(0, 0, 0),
                            blur_value=1,
                            rain_type='drizzle',
                            p=0.05
                        ),
                        A.GaussNoise(
                            p=0.05, var_limit=(100.0, 500.0), mean=100.0
                        ),
                        A.PlasmaShadow(
                            p=0.05, shadow_intensity_range=(0.3, 0.7), plasma_size=256, roughness=3.0
                        ),
                        A.Posterize(
                            p=0.05, num_bits=4
                        ),
                        A.Defocus(radius=(3, 10), alias_blur=(0.1, 0.5), p=0.05),
                        A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.05),
                        A.ColorJitter(
                            brightness=(0.8, 1.2),
                            contrast=(0.8, 1.2),
                            saturation=(0.8, 1.2),
                            hue=(-0.5, 0.5),
                            p=0.05
                        ),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.05),

                        # Quality augmentations.
                        A.OneOf(
                            [
                                A.ImageCompression(quality_range=(10, 90), compression_type="jpeg"),
                                A.ImageCompression(quality_range=(10, 90), compression_type="webp"),
                            ],
                            p=0.1
                        ),
                        A.Downscale(
                            scale_range=(0.25, 0.25),
                            interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR},
                            p=0.05
                        ),
                    ],
                )
            ))

            # Geometric transformations.
            add_augmentations.append(AlbumentationWrapper(
                A.Compose(
                    [
                        A.HorizontalFlip(p=0.3),
                    ],
                )
            ))

            # Rect unsafe transformations - apply only for images that have segmentation annotations.
            add_augmentations.append(RectUnsafeTransformationsWrapper(
                AlbumentationWrapper(
                    A.Compose(
                        [
                            A.SafeRotate(
                                limit=(-70, 70),
                                border_mode=cv2.BORDER_CONSTANT,
                                interpolation=cv2.INTER_CUBIC,
                                fill=(0, 0, 0),
                                p=0.3
                            ),
                            A.GridElasticDeform(num_grid_xy=(8, 8), magnitude=10, p=0.1),
                        ],
                    )
                )
            ))

        add_augmentations.append(ClosingAugmentation(add_control_bbox=self._add_control_bbox))
        add_augmentations.append(nx.dataset.utils.FilterInvisibleBboxes())
        transforms.insert(0, ultralytics.data.augment.Compose(add_augmentations))

        return transforms

    """
    get_image_and_label should return:
    im_file
    cls
    img: numpy.ndarray(h, w, c), cv2 image
    ori_shape: (h, w)
    resized_shape: (h, w)
    ratio_pad: (1.0, 1.0)
    instances: Instances
    ---------------------------------------------
    __getitems__ should return (h=640, w=481):
    im_file
    img: torch.tensor
    ori_shape: (640, 481)
    resized_shape: (640, 640)
    shape: (640, 481)
    batch_idx: : tensor([0.])
    cls: tensor([[0.]])
    bboxes: tensor([[0.6947, 0.5982, 0.4784, 0.7176]])
    """
    def __getitems__(self, indexes):
        # Implement __getitems__ for apply batch transformations.
        datas = [self.get_image_and_label(index) for index in indexes]
        # Apply batch transformations to "img".
        transformed_datas = []
        for data_i, data in enumerate(datas):
            assert 'loaded_segments' in data, "loaded_segments should be filled before transformations"

            transformed_data = self.transforms(data)

            assert 'bboxes' in transformed_data, "bboxes should be filled after all transformations"

            if self._add_control_bbox:
                # loaded_segments can disappears after some specific unltralytics YOLO transformations (mosaic).
                if "loaded_segments" in transformed_data:
                    if abs(transformed_data['cls'][-1][0].item() + 1) < 0.00001:  # < last cls is -1
                        # Remove control bbox (added in ClosingAugmentation) and adapt segments by it
                        transformed_data['cls'] = transformed_data['cls'][:-1]
                        last_bbox = transformed_data['bboxes'][-1].numpy().tolist()
                        new_center_x, new_center_y, new_width, new_height = last_bbox
                        x1 = new_center_x - new_width / 2
                        y1 = new_center_y - new_height / 2
                        segments = transformed_data['loaded_segments']
                        transformed_segments = []
                        for segment in segments:
                            transformed_segments.append(
                                [[p[0] * new_width + x1, p[1] * new_height + y1] for p in segment.polygon]
                            )
                        transformed_data['loaded_segments'] = transformed_segments
                else:
                    # Remove all -1 classes and bboxes for these classes.
                    cls_tensor = transformed_data['cls']
                    bboxes_tensor = transformed_data['bboxes']
                    cls = cls_tensor.numpy().tolist()
                    bboxes = bboxes_tensor.numpy().tolist()
                    filtered_cls = []
                    filtered_bboxes = []
                    for cls_index, c in enumerate(cls):
                        if abs(c[0] + 1) < 0.00001:
                            pass
                        else:
                            filtered_cls.append(c)
                            filtered_bboxes.append(bboxes[cls_index])
                    assert len(filtered_cls) == len(filtered_bboxes)
                    if len(filtered_cls) > 0:
                        transformed_data['cls'] = torch.tensor(filtered_cls, dtype=cls_tensor.dtype)
                        transformed_data['bboxes'] = torch.tensor(filtered_bboxes, dtype=bboxes_tensor.dtype)
                    else:
                        transformed_data['cls'] = torch.empty((0, 1), dtype=torch.float32)
                        transformed_data['bboxes'] = torch.empty((0, 4), dtype=torch.float32)

            transformed_datas.append(transformed_data)

        return transformed_datas
