import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two sets of bounding boxes.
    """

    bb_test = np.expand_dims(bb_test, 1)
    bb_gt = np.expand_dims(bb_gt, 0)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)

    intersection = w * h

    area_test = (
        (bb_test[..., 2] - bb_test[..., 0]) *
        (bb_test[..., 3] - bb_test[..., 1])
    )

    area_gt = (
        (bb_gt[..., 2] - bb_gt[..., 0]) *
        (bb_gt[..., 3] - bb_gt[..., 1])
    )

    union = area_test + area_gt - intersection

    return intersection / (union + 1e-6)


def convert_bbox_to_z(bbox):
    """
    Convert [x1,y1,x2,y2] to [center_x, center_y, area, aspect_ratio].
    """

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0

    area = w * h
    aspect_ratio = w / float(h + 1e-6)

    return np.array([x, y, area, aspect_ratio]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Convert Kalman state back to [x1,y1,x2,y2].
    """

    w = np.sqrt(abs(x[2] * x[3]))
    h = x[2] / (w + 1e-6)

    if score is None:
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0
        ]).reshape((1, 4))

    return np.array([
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
        score
    ]).reshape((1, 5))


class KalmanBoxTracker:
    """
    Represents the internal state of one tracked object.
    """

    count = 0

    def __init__(self, bbox):

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count + 1
        KalmanBoxTracker.count += 1

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):

        self.time_since_update = 0
        self.history = []

        self.hits += 1
        self.hit_streak += 1

        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):

        if (
            self.kf.x[6] + self.kf.x[2] <= 0
        ):
            self.kf.x[6] *= 0.0

        self.kf.predict()

        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1

        self.history.append(
            convert_x_to_bbox(self.kf.x)
        )

        return self.history[-1]

    def get_state(self):

        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(
    detections,
    trackers,
    iou_threshold=0.3
):
    """
    Assign detections to existing trackers using IoU.
    """

    if len(trackers) == 0:

        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int)
        )

    if len(detections) == 0:

        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(len(trackers))
        )

    iou_matrix = iou_batch(
        detections,
        trackers
    )

    row_ind, col_ind = linear_sum_assignment(
        -iou_matrix
    )

    matched_indices = np.column_stack(
        (row_ind, col_ind)
    )

    unmatched_detections = []

    for d in range(len(detections)):

        if d not in matched_indices[:, 0]:

            unmatched_detections.append(d)

    unmatched_trackers = []

    for t in range(len(trackers)):

        if t not in matched_indices[:, 1]:

            unmatched_trackers.append(t)

    matches = []

    for m in matched_indices:

        if iou_matrix[m[0], m[1]] < iou_threshold:

            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])

        else:

            matches.append(m.reshape(1, 2))

    if len(matches) == 0:

        matches = np.empty((0, 2), dtype=int)

    else:

        matches = np.concatenate(matches, axis=0)

    return (
        matches,
        np.array(unmatched_detections, dtype=int),
        np.array(unmatched_trackers, dtype=int)
    )


class Sort:

    def __init__(
        self,
        max_age=1,
        min_hits=3,
        iou_threshold=0.3
    ):

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []

        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):

        self.frame_count += 1

        trks = np.zeros(
            (len(self.trackers), 5)
        )

        to_del = []

        ret = []

        for t, tracker in enumerate(self.trackers):

            pos = tracker.predict()[0]

            trks[t, :] = [
                pos[0],
                pos[1],
                pos[2],
                pos[3],
                0
            ]

            if np.any(np.isnan(pos)):

                to_del.append(t)

        trks = np.ma.compress_rows(
            np.ma.masked_invalid(trks)
        )

        for t in reversed(to_del):

            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = (
            associate_detections_to_trackers(
                dets,
                trks,
                self.iou_threshold
            )
        )

        for m in matched:

            self.trackers[m[1]].update(
                dets[m[0], :]
            )

        for i in unmatched_dets:

            tracker = KalmanBoxTracker(
                dets[i, :]
            )

            self.trackers.append(tracker)

        i = len(self.trackers)

        for tracker in reversed(self.trackers):

            d = tracker.get_state()[0]

            if (
                tracker.time_since_update < 1
                and (
                    tracker.hit_streak >= self.min_hits
                    or self.frame_count <= self.min_hits
                )
            ):

                ret.append(
                    np.concatenate(
                        (d, [tracker.id])
                    )
                )

            i -= 1

            if (
                tracker.time_since_update >
                self.max_age
            ):

                self.trackers.pop(i)

        if len(ret) > 0:

            return np.stack(ret)

        return np.empty((0, 5))