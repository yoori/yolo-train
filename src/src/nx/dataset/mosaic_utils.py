import typing
import enum
import random
import copy
import torch
import torchvision.transforms.v2

import cv2

import nx.dataset.segment_utils


def create_simple_mosaic(
    tensor: torch.Tensor,
    segments: typing.List[nx.dataset.utils.Segment],
    x_left = 0,
    y_top = 0,
    x_right = 0,
    y_bottom = 0,
    result_width = 640,
    result_height = 640,
    gap_x = 10,
    gap_y = 10,
) -> typing.Tuple[torch.Tensor, typing.List[nx.dataset.utils.Segment]]:
    # tensor have shape [3, width, height]
    c, height, width = tensor.shape
    result = torch.zeros(
        3, result_height, result_width,
        dtype=tensor.dtype
    ).to(device=tensor.device)
    print("width = " + str(width))
    print("height = " + str(height))
    print("tensor.shape = " + str(tensor.shape))
    print("result.shape = " + str(result.shape))
    x_step = width + gap_x
    y_step = height + gap_y
    x_steps = int((result_width - x_left - x_right) / x_step)
    y_steps = int((result_height - y_top - y_bottom) / y_step)
    result_segments = []
    for x_i in range(x_steps):
        for y_i in range(y_steps):
            paste_x = x_left + x_i*x_step
            paste_y = y_top + y_i*y_step
            result[
                :,
                paste_y:paste_y + height,
                paste_x:paste_x + width
            ] = tensor
            for s in segments:
                s = copy.deepcopy(s)
                if s.bbox is not None:
                    s.bbox = (
                        min(max((s.bbox[0] * width + paste_x) / result_width, 0), 1),
                        min(max((s.bbox[1] * height + paste_y) / result_height, 0), 1),
                        min(max((s.bbox[2] * width + paste_x) / result_width, 0), 1),
                        min(max((s.bbox[3] * height + paste_y) / result_height, 0), 1),
                    )
                else:
                    s.polygon = [
                       (
                           min(max((point[0] * width + paste_x) / result_width, 0), 1),
                           min(max((point[1] * height + paste_y) / result_height, 0), 1),
                       ) for point in s.polygon
                    ]
                result_segments.append(s)
    return (result, result_segments)


def create_overlayed_mosaic(
    tensor: torch.Tensor,
    segments: typing.List[nx.dataset.utils.Segment],  # < segments relative tensor.
    x_left = 0,
    y_top = 0,
    x_right = 0,
    y_bottom = 0,
    result_width = 640,
    result_height = 640,
    gap_x = 0,
    gap_y = 0,
) -> torch.Tensor:
    _, height, width = tensor.shape
    tensor1, segments1 = create_simple_mosaic(
        tensor,
        segments,
        x_left=x_left,
        y_top=y_top,
        x_right=x_right,
        y_bottom=y_bottom,
        result_width=result_width,
        result_height=result_height,
        gap_x=gap_x,
        gap_y=gap_y,
    )
    tensor2, segments2 = create_simple_mosaic(
        tensor,
        segments,
        x_left=x_left + (width + gap_x) // 2,
        y_top=y_top + (height + gap_y) // 2,
        x_right=x_right,
        y_bottom=y_bottom,
        result_width=result_width,
        result_height=result_height,
        gap_x=gap_x,
        gap_y=gap_y,
    )
    return (tensor1 / 2 + tensor2 / 2, segments1 + segments2)


class ObjectMosaicMode(enum.Enum):
    NONE = 0,
    SIMPLE = 1,
    WITH_OVERLAY = 2


def cut_and_paste_object_as_mosaic(
    tensor: torch.Tensor,
    safe_for_cut_bboxes: typing.List[typing.Tuple[float, float, float, float]],
    segments: typing.List[nx.dataset.utils.Segment],
    result_width = 640,
    result_height = 640,
    mode = ObjectMosaicMode.SIMPLE,
    gap_x = 0,
    gap_y = 0,
):
    min_result_size = 30
    _, height, width = tensor.shape  # < CHW
    cut_box = random.choice(safe_for_cut_bboxes)
    # don't decrease object size if it already have some size less then 30 px. 
    min_ratio = min(max(min_result_size / width, min_result_size / height), 1)
    max_ratio = max(min(result_width / 2 / width, result_height / 2 / height), min_ratio)
    print("SELECT ratio in [" + str(min_ratio) + ", " + str(max_ratio) + "]")
    ratio = random.uniform(min_ratio, max_ratio)
    print("RESULT RATIO: " + str(ratio))
    # resize for ratio
    tensor = torchvision.transforms.functional.resize(
        tensor,
        size=(int(height * ratio), int(width * ratio)),
    )
    _, height, width = tensor.shape
    segments = nx.dataset.segment_utils.adapt_segments_for_cut(cut_box, segments)
    if mode == ObjectMosaicMode.SIMPLE:
        image_tensor, segments = create_simple_mosaic(
            tensor,
            segments,
            x_left=min(result_width - width, random.randint(0, width // 2)),
            y_top=min(result_height - height, random.randint(0, height // 2)),
            gap_x=gap_x,
            gap_y=gap_y,
        )
    else:
        image_tensor, segments = create_overlayed_mosaic(
            tensor,
            segments,
            x_left=min(result_width - width, random.randint(0, width // 2)),
            y_top=min(result_height - height, random.randint(0, height // 2)),
            gap_x=gap_x,
            gap_y=gap_y,
        )
    return image_tensor, segments
    

if __name__ == "__main__":
    random.seed()
    input_image = cv2.imread('TT/t_in2.jpg')
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(input_image)

    #print("input_tensor.dtype(0) = " + str(input_tensor.dtype))
    input_tensor = input_tensor.permute(2, 0, 1)
    #print("input_tensor.shape(P) = " + str(input_tensor.shape))
    #print("input_tensor.dtype(P) = " + str(input_tensor.dtype))

    print("input_tensor.shape(0) = " + str(input_tensor.shape))

    # Resize
    input_tensor = torchvision.transforms.v2.RandomResizedCrop(
        size=(52, 91), antialias=True
    )(input_tensor)

    print("input_tensor.shape(1) = " + str(input_tensor.shape))

    # Create mosaic
    image_tensor, _ = cut_and_paste_object_as_mosaic(
        input_tensor,
        [(0, 0, 1, 1)],
        [],
        result_width=640,
        result_height=640,
        mode=ObjectMosaicMode.SIMPLE,
    )

    # Save
    image = image_tensor.detach().cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image[:, :, ::-1]
    cv2.imwrite('TT/t_res_simple.jpg', image)

    image_tensor, _ = cut_and_paste_object_as_mosaic(
        input_tensor,
        [(0, 0, 1, 1)],
        [],
        result_width=640,
        result_height=640,
        mode=ObjectMosaicMode.WITH_OVERLAY,
    )

    # Save
    image = image_tensor.detach().cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image[:, :, ::-1]
    cv2.imwrite('TT/t_res_overlay.jpg', image)
