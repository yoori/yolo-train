import nx.dataset.utils


MUL = 1000


def cross_test():
    poly = [[0, 0], [1, 0], [0, 1]]
    bbox = (0, 0, 0.5, 0.5)
    poly2 = [[0, 0], [1, 0], [0.5, 0.5]]
    bbox2 = (0, 0, 0.25, 0.25)
    p1 = nx.dataset.utils.blocked_area_percentage(
        nx.dataset.utils.Segment(label="0", polygon=poly),
        [nx.dataset.utils.Segment(label="0", bbox=bbox)],
    )
    assert p1 == 0.25 / 0.5
    p2 = nx.dataset.utils.blocked_area_percentage(
        nx.dataset.utils.Segment(label="0", bbox=bbox),
        [nx.dataset.utils.Segment(label="0", polygon=poly)],
    )
    assert p2 == 1
    p3 = nx.dataset.utils.blocked_area_percentage(
        nx.dataset.utils.Segment(label="0", polygon=poly2),
        [nx.dataset.utils.Segment(label="0", polygon=poly)],
    )
    assert p3 == 0.25 / 0.25
    p4 = nx.dataset.utils.blocked_area_percentage(
        nx.dataset.utils.Segment(label="0", bbox=bbox),
        [nx.dataset.utils.Segment(label="0", bbox=bbox2)],
    )
    assert p4 == 0.25 * 0.25 / 0.25


def find_blocked_segments_test():
    all_polygons = [
        nx.dataset.utils.Segment(
            label="0",
            bbox=(1, 4, 6, 8),
        ),
        nx.dataset.utils.Segment(
            label="0",
            bbox=(5, 4, 8, 13),
        ),
        nx.dataset.utils.Segment(
            label="0",
            bbox=(6, 6, 11, 11),
        ),
        nx.dataset.utils.Segment(
            label="0",
            bbox=(9, 9, 14, 14),
        ),
        nx.dataset.utils.Segment(
            label="0",
            bbox=(12, 7, 15, 16),
        ),
        nx.dataset.utils.Segment(
            label="0",
            bbox=(14, 12, 19, 16),
        ),
    ]
    non_blocked_segments, blocked_segments = nx.dataset.utils.find_blocked_segments(
        all_polygons,
        [
            nx.dataset.utils.Segment(
                label="0",
                bbox=(8, 8, 12, 12),
            ),
        ],
    )
    assert len(non_blocked_segments) + len(blocked_segments) == len(all_polygons) + 1
    assert len(non_blocked_segments) == 2
    assert non_blocked_segments[0].bbox[0] == 1
    assert non_blocked_segments[1].bbox[0] == 14


if __name__ == '__main__':
    cross_test()
    find_blocked_segments_test()
