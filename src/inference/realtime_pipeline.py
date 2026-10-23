"""Latest-frame inference with timestamped results and bounded motion tracking.

Models see clean frames only. One coordinator bounds outstanding work, with
an optional two-thread detector pool on CPU. Cached geometry is motion-compensated briefly;
untrackable or old results are hidden rather than drawn at obsolete positions.
"""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    frame_id: int
    captured_at: float
    completed_at: float
    frame: np.ndarray
    actual_deg: float
    steering_deg: Optional[float]
    lanes: list
    objects: list
    timings: dict
    errors: dict


class RealtimePipeline:
    """One coordinator and replaceable input slot; no accumulating frame queue."""

    def __init__(self, lane_detector, object_detector, steering: Callable, parallel_detectors=False):
        self.lane_detector = lane_detector
        self.object_detector = object_detector
        self.steering = steering
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='AutoDrive-perception') if parallel_detectors else None
        self._condition = threading.Condition()
        self._pending = None
        self._result = None
        self._stopping = False
        self._last_error_log = {}
        self._thread = threading.Thread(target=self._run, name='AutoDrive-inference', daemon=True)
        self._thread.start()

    def submit(self, frame_id, frame, actual_deg, lanes=True, objects=True):
        with self._condition:
            self._pending = (frame_id, time.perf_counter(), frame.copy(), actual_deg, lanes, objects)
            self._condition.notify()

    def latest(self):
        with self._condition:
            return self._result

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._stopping)
                if self._stopping:
                    return
                frame_id, captured_at, frame, actual, use_lanes, use_objects = self._pending
                self._pending = None
            timings, errors = {}, {}

            def measured(name, operation, fallback):
                started = time.perf_counter()
                try:
                    return operation()
                except Exception as exc:
                    errors[name] = str(exc)
                    now = time.perf_counter()
                    if now - self._last_error_log.get(name, 0) >= 5:
                        LOGGER.exception('%s inference failed on frame %s', name, frame_id)
                        self._last_error_log[name] = now
                    return fallback
                finally:
                    timings[name] = (time.perf_counter() - started) * 1000

            angle = measured('steering', lambda: float(self.steering(frame)), None)
            def lane_pass():
                return measured('lane', lambda: self.lane_detector.detect(frame)[1], {}) if use_lanes and self.lane_detector else {}

            def object_pass():
                return measured('object', lambda: self.object_detector.detect(frame)[1], {}) if use_objects and self.object_detector else {}

            if self._pool:
                lane_job = self._pool.submit(lane_pass)
                object_job = self._pool.submit(object_pass)
                lane_info, object_info = lane_job.result(), object_job.result()
            else:
                lane_info, object_info = lane_pass(), object_pass()
            timings['total'] = (time.perf_counter() - captured_at) * 1000
            result = Prediction(frame_id, captured_at, time.perf_counter(), frame, actual, angle,
                                lane_info.get('lanes', []), object_info.get('objects', []), timings, errors)
            with self._condition:
                self._result = result

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=10)
        if not self._thread.is_alive() and self._pool:
            self._pool.shutdown(wait=True)
        return not self._thread.is_alive()


def align_prediction(result: Optional[Prediction], frame, frame_id, max_age=0.75, max_frame_gap=15):
    """Return current-frame geometry, or hide it when tracking is unreliable.

    Lane motion uses robust road-region optical flow. Each object uses flow
    points inside its own box. This is short-term tracking, not a new detection.
    Source age and frame gap are bounded; seeks never reuse old results.
    """
    info = {'num_lanes': 0, 'lanes': [], 'num_objects': 0, 'objects': [],
            'state': 'WARMING UP', 'age_ms': None, 'source_frame': None}
    if result is None:
        return info
    age = time.perf_counter() - result.captured_at
    gap = frame_id - result.frame_id
    info.update(age_ms=age * 1000, source_frame=result.frame_id)
    if result.errors:
        info['state'] = 'INFERENCE ERROR'
    if gap < 0 or gap > max_frame_gap or age > max_age:
        info['state'] = 'WAITING FOR FRESH RESULT'
        return info
    if gap == 0:
        lanes, objects = result.lanes, result.objects
        state = 'DETECTED'
    else:
        small_size = (320, 180)
        old_gray = cv2.cvtColor(cv2.resize(result.frame, small_size), cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(cv2.resize(frame, small_size), cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(old_gray, maxCorners=300, qualityLevel=.01, minDistance=5)
        if points is None or len(points) < 8:
            info['state'] = 'TRACKING UNAVAILABLE'
            return info
        moved, valid, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, points, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, .03))
        if moved is None:
            info['state'] = 'TRACKING UNAVAILABLE'
            return info
        # Reject flow with poor appearance match or inconsistent reverse flow.
        back, back_valid, _ = cv2.calcOpticalFlowPyrLK(new_gray, old_gray, moved, None,
                                                     winSize=(21, 21), maxLevel=3)
        if back is None:
            return info
        keep = valid.ravel().astype(bool) & back_valid.ravel().astype(bool)
        keep &= error.ravel() < 25
        keep &= np.linalg.norm(points.reshape(-1, 2) - back.reshape(-1, 2), axis=1) < 1.5
        src, dst = points.reshape(-1, 2)[keep], moved.reshape(-1, 2)[keep]
        scale = np.array([frame.shape[1] / 320, frame.shape[0] / 180], dtype=np.float32)
        lanes, objects = [], []
        road = src[:, 1] > 180 * .40
        if road.sum() >= 8:
            transform, inliers = cv2.estimateAffinePartial2D(src[road] * scale, dst[road] * scale,
                                      method=cv2.RANSAC, ransacReprojThreshold=3)
            if transform is not None and inliers is not None and inliers.mean() >= .5:
                zoom = float(np.hypot(transform[0, 0], transform[0, 1]))
                if .85 <= zoom <= 1.15:
                    for lane in result.lanes:
                        contour = cv2.transform(np.asarray(lane, np.float32).reshape(-1, 1, 2), transform)
                        lanes.append(contour.astype(np.int32))
        for obj in result.objects:
            x1, y1, x2, y2 = np.asarray(obj['box'], dtype=np.float32) / np.tile(scale, 2)
            inside = (src[:, 0] >= x1) & (src[:, 0] <= x2) & (src[:, 1] >= y1) & (src[:, 1] <= y2)
            if inside.sum() < 3:
                continue
            displacement = (dst[inside] - src[inside]) * scale
            motion = np.median(displacement, axis=0)
            if np.median(np.linalg.norm(displacement - motion, axis=1)) > 8:
                continue
            box = np.asarray(obj['box'], dtype=float) + np.tile(motion, 2)
            box[[0, 2]] = np.clip(box[[0, 2]], 0, frame.shape[1] - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, frame.shape[0] - 1)
            if box[2] > box[0] and box[3] > box[1]:
                objects.append(dict(obj, box=box.astype(int).tolist()))
        state = 'TRACKED'
    info.update(lanes=lanes, num_lanes=len(lanes), objects=objects,
                num_objects=len(objects), state='INFERENCE ERROR' if result.errors else state)
    return info
