"""
Base script for develop and test new dali trasformation (in dali pipeline).
Read annotated images, pass annotations as keypoints into transformation and
save trasformed images and keypoints as annotated images.
"""
import typing
import numpy as np
import pathlib
import argparse
import nvidia.dali as dali
import kornia as K
import torch
import nvidia.dali
import nvidia.dali.plugin.pytorch as dali_plugin_pytorch
import cv2

import nx.dataset.utils
from nx.dataset.dali_kornia_dataset import DaliKorniaDataset, resize_and_crop, rotate


def get_data(keypoints):
    return [np.float32(x) for x in keypoints]


def get_rel_start_by_bboxes(bboxes):
    return [np.float32((bbox[0], bbox[1])) for bbox in bboxes]


def get_rel_end_by_bboxes(bboxes):
    return [np.float32((bbox[2], bbox[3])) for bbox in bboxes]


@dali.pipeline_def(exec_dynamic=True)
def dali_pipeline(files, keypoints = None, cut_regions = None):
    result_size = 640

    jpegs, labels = dali.fn.readers.file(
        files = files,
        labels = list(range(len(files))),
    )
    images = dali.fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=dali.types.DALIImageType.BGR  # < By fact, output will be RGB
    )
    process_keypoints = nvidia.dali.fn.external_source(
        source=lambda: get_data(keypoints),
        dtype=nvidia.dali.types.FLOAT
    )

    cut_rel_start = nvidia.dali.fn.external_source(
        source=lambda: get_rel_start_by_bboxes(cut_regions),
        dtype=nvidia.dali.types.FLOAT
    )

    cut_rel_end = nvidia.dali.fn.external_source(
        source=lambda: get_rel_end_by_bboxes(cut_regions),
        dtype=nvidia.dali.types.FLOAT
    )

    images = nvidia.dali.fn.slice(images, rel_start=cut_rel_start, rel_end=cut_rel_end)

    # Resize for return tensor list with equal shapes, otherwise batch processing is impossible.
    images, process_keypoints = resize_and_crop(images, process_keypoints, size=result_size)
    images = dali.fn.transpose(images, perm=[2, 0, 1])
    images = dali.fn.cast(images, dtype=dali.types.DALIDataType.UINT8)

    return images, labels, process_keypoints  # < labels allow to map result to input files.


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image-root', help = 'Load images from this directory and pass as single batch', required = True)
    parser.add_argument('--save-root', help = '', default = '')
    args = parser.parse_args()

    images: typing.List[str] = []
    image_segments: typing.List[typing.List[nx.dataset.utils.Segment]] = []
    images_keypoints: typing.List[typing.List[typing.Tuple[float, float]]] = []
    for image_file in pathlib.Path(args.image_root).glob('**/*.jpg'):
        images.append(str(image_file))
        ann_file = image_file.parent / (image_file.stem + ".json")
        segments = []
        file_keypoints = []
        if ann_file.is_file():
            segments, _, _ = nx.dataset.utils.read_labelme_annotations(ann_file)
            segments = [segment.convert_to_polygon_segment() for segment in segments]
            file_keypoints = DaliKorniaDataset.polygon_segments_to_keypoints(segments)
        image_segments.append(segments)
        images_keypoints.append(file_keypoints)

    assert len(images) == len(images_keypoints)

    pipe = dali_pipeline(
        files=images,
        batch_size=len(images),
        num_threads=1,
        device_id=0,
        keypoints=images_keypoints,
        cut_regions=[
            [0.25, 0.25, 0.75, 0.75] for x in range(len(images))
        ]
    )
    pipe.build()
    image_batch, label_batch, keypoints_batch = pipe.run()
    images_dali_tensor = image_batch.as_tensor()
    labels_dali_tensor = label_batch.as_tensor()

    images_torch_tensor = torch.empty(images_dali_tensor.shape(), dtype=torch.uint8, device=torch.device('cuda:0'))
    dali_plugin_pytorch.feed_ndarray(images_dali_tensor, images_torch_tensor)
    labels_torch_tensor = torch.empty(labels_dali_tensor.shape(), dtype=torch.int32, device=torch.device('cpu'))
    dali_plugin_pytorch.feed_ndarray(labels_dali_tensor, labels_torch_tensor)

    keypoints = []  # < array of keypoints arrays (array per image)
    for keypoints_dali_tensor in keypoints_batch:  # keypoints_batch is list of dali tensors.
        keyword_torch_tensor = torch.empty(keypoints_dali_tensor.shape(), dtype=torch.float32, device=torch.device('cpu'))
        dali.plugin.pytorch.feed_ndarray(keypoints_dali_tensor, keyword_torch_tensor)
        keypoints.append(keyword_torch_tensor.numpy().tolist())

    for image_tensor, file_index_tensor, item_keypoints in zip(
        images_torch_tensor, labels_torch_tensor, keypoints
    ):
        source_file = pathlib.Path(images[file_index_tensor.item()])
        segments = image_segments[file_index_tensor.item()]
        transformed_segments = DaliKorniaDataset.keypoints_to_segments(item_keypoints, segments)
        save_image = K.tensor_to_image(image_tensor)
        save_image = save_image.astype(np.uint8).copy()
        h, w, _ = save_image.shape
        cv2.imwrite(pathlib.Path(args.save_root) / source_file.name, save_image)
        nx.dataset.utils.write_labelme_annotations(
            pathlib.Path(args.save_root) / (source_file.stem + ".json"),
            segments=transformed_segments,
            image_path=source_file.name,
            image_width=w,
            image_height=h,
        )


if __name__ == "__main__":
    main()
