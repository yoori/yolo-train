import typing
import copy

import nx.dataset.utils


def segment_bbox_is_crossed(
    segment: nx.dataset.utils.Segment,
    segments: typing.List[nx.dataset.utils.Segment]
) -> bool:
    for check_segment in segments:
        if segment.iou(check_segment) > 0:
            return True

    return False


def extend_bbox(bbox: typing.Tuple[float, float, float, float], percent=0.1):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    new_left = max(bbox[0] - width * percent / 2, 0)
    new_right = min(bbox[2] + width * percent / 2, 1)
    new_top = max(bbox[1] - height * percent / 2, 0)
    new_bottom = min(bbox[3] + height * percent / 2, 1)
    return (new_left, new_top, new_right, new_bottom)


def adapt_segments_for_cut(
    cut_bbox: typing.Tuple[float, float, float, float],
    segments: typing.List[nx.dataset.utils.Segment]
) -> typing.List[nx.dataset.utils.Segment]:
    result = []
    for s in segments:
        s = copy.deepcopy(s)
        if s.bbox is not None:
            s.bbox = (
                min(max((s.bbox[0] - cut_bbox[0]) / (cut_bbox[2] - cut_bbox[0]), 0), 1),
                min(max((s.bbox[1] - cut_bbox[1]) / (cut_bbox[3] - cut_bbox[1]), 0), 1),
                min(max((s.bbox[2] - cut_bbox[0]) / (cut_bbox[2] - cut_bbox[0]), 0), 1),
                min(max((s.bbox[3] - cut_bbox[1]) / (cut_bbox[3] - cut_bbox[1]), 0), 1),
            )
        else:
            s.polygon = [
                (
                    min(max((point[0] - cut_bbox[0]) / (cut_bbox[2] - cut_bbox[0]), 0), 1),
                    min(max((point[1] - cut_bbox[1]) / (cut_bbox[3] - cut_bbox[1]), 0), 1)
                ) for point in s.polygon
            ]
        result_bbox = s.get_bbox()
        if result_bbox[3] - result_bbox[1] > 0 and result_bbox[2] - result_bbox[0] > 0:
            result.append(s)
    return result
            
                
def select_boxes_safe_for_cut(
    segments: typing.List[nx.dataset.utils.Segment],
    image_width = 640,
    image_height = 640,
    min_width = 30,
    min_height = 30,
) -> typing.List[
    typing.Tuple[float, float, float, float]
]:
    extend_coefs = [1.0, 0.5, 0.2, 0.1]
    # Segment that can be cutted
    result = []
    for segment_i, s in enumerate(segments):
        original_bbox = s.get_bbox()
        if (
            (original_bbox[2] - original_bbox[0]) * image_width > min_width and
            (original_bbox[3] - original_bbox[1]) * image_height > min_height and
            not segment_bbox_is_crossed(s, segments[: segment_i] + segments[segment_i + 1:])
        ):
            appended = False
            for extend_coef in extend_coefs:
                # Try extend bbox for save context
                extended_bbox = extend_bbox(original_bbox, percent=extend_coef)
                if not segment_bbox_is_crossed(
                    nx.dataset.utils.Segment(bbox=extended_bbox),
                    segments[: segment_i] + segments[segment_i + 1:]
                ):
                    result.append(extended_bbox)
                    appended = True
            if not appended:
                result.append(original_bbox)
    return result
