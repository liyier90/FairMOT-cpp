"""Utility functions for reading MOT annotation or detection file."""

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, NamedTuple, Tuple, Union

import numpy as np


def read_results(
    results_path: Path, is_gt: bool = False, is_ignore: bool = False
) -> DefaultDict[int, List[Tuple[Tuple[float, ...], int, Union[float, int]]]]:
    """Reads ground truth annotations files and detection results files.

    Args:
        results_path (Path): Path to annotation or detection results file.
        is_gt (bool): Flag to indicate if we are parsing selected ground truth
            annotations.
        is_ignore (bool): Flag to indicate if we are parsing ignored ground
            truth annotations.

    Returns:
        (DefaultDict[int, List[Tuple[Tuple[float, ...], int, Union[float, int]]]]):
        A dictionary mapping the frame ID to a list of detections which are
        represented by a tuple containing:
        - tlwh (Tuple[float, ...]): Bounding box in [t, l, w, h] format where
            (t, l) is the top-left corner, w is width, and h is height.
        - target_id (int): The track ID of the detection.
        - score (Union[float, int]): For ground truth, this is a flag
            indicating if the detection should be ignored (0) or included (1).
            For detection, this is the detection confidence score.
    """
    return _read_mot_results(results_path, is_gt, is_ignore)


def unzip_objs(
    objs: List[Tuple[Tuple[float, ...], int, Union[float, int]]]
) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[Union[float, int], ...]]:
    """Unzips a list of tuples into three separate lists.

    Args:
        objs (List[Tuple[Tuple[float, ...], int, Union[float, int]]]): List of
            entries parsed from either a MOT annotation file or a result file.
            Each entry contains:
            - tlwh (Tuple[float, ...]): Bounding box in [t, l, w, h] format
                where (t, l) is the top-left corner, w is width, and h is
                height.
            - target_id (int): The track ID of the detection.
            - score (Union[float, int]): For ground truth, this is a flag
                indicating if the detection should be ignored (0) or
                included (1). For detection, this is the detection confidence
                score.

    Returns:
        Tuple[np.ndarray, Tuple[int, ...], Tuple[Union[float, int], ...]]:
        Unzips the given list into 3 separate components:
        - tlwh (np.ndarray): Bounding box coordinates converted to a numpy
            array.
        - target_id
        - score
    """
    if objs:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = (), (), ()
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores


class MOTEntry(NamedTuple):
    """A single entry of either MOT challenge ground truth entry or detection
    result.

    Attributes:
        frame_id (int): Frame number of the video sequence.
        target_id (int): Tracking ID of the detection. Corresponds to "Identity
            number" in Table 5 in "MOT16: A Benchmark for Multi-Object
            Tracking".
        tlwh (Tuple[float, ...]): Bounding box with the format [t, l, w, h]
            where (t, l) is the top-left corner, w is width, and h is height.
        score (Union[float, int]): For ground truths, this acts as a flag to
            indicate if the entry is considered (1) or ignored (0). For
            detection results, this is the detection confidence score.
        label (int): Object class ID. Only used for ground truths, defaults to
            `-1` for detection results.
        vis_ratio (float): Visibility ratio between [0, 1] to indicate how much
            of that object is visible. Only used for ground truths, defaults to
            `-1.0` for detection results.
    """

    frame_id: int
    target_id: int
    tlwh: Tuple[float, ...]
    score: Union[float, int]
    label: int
    vis_ratio: float

    @classmethod
    def create(cls, line_parts: List[str], is_detection: bool) -> "MOTEntry":
        """Constructs a `MOTEntry` for either a ground truth annotation or a
        detection result.

        Args:
            line_parts (List[str]): Parts of a line from either an annotation
                file or detection result file after it's split by comma (,).
            is_detection (bool): Flag to indicate if `line_parts` came from an
                annotation file or a detection result file. Uses placeholder
                values for `label` and `vis_ratio` if it's from a detection
                result file.
        """
        if is_detection:
            return cls(
                int(line_parts[0]),
                int(line_parts[1]),
                tuple(map(float, line_parts[2:6])),
                float(line_parts[6]),
                -1,
                -1.0,
            )
        return cls(
            int(line_parts[0]),
            int(line_parts[1]),
            tuple(map(float, line_parts[2:6])),
            int(line_parts[6]),
            int(line_parts[7]),
            float(line_parts[8]),
        )


def _read_mot_results(
    results_path: Path, is_gt: bool, is_ignore: bool
) -> DefaultDict[int, List[Tuple[Tuple[float, ...], int, Union[float, int]]]]:
    """Reads ground truth annotations files and detection results files for the
    MOT Challenge.

    Args:
        results_path (Path): Path to annotation or detection results file.
        is_gt (bool): Flag to indicate if we are parsing selected ground truth
            annotations.
        is_ignore (bool): Flag to indicate if we are parsing ignored ground
            truth annotations.

    Returns:
        (DefaultDict[int, List[Tuple[Tuple[float, ...], int, Union[float, int]]]]):
        A dictionary mapping the frame ID to a list of detections which are
        represented by a tuple containing:
        - tlwh (Tuple[float, ...]): Bounding box in [t, l, w, h] format where
            (t, l) is the top-left corner, w is width, and h is height.
        - target_id (int): The track ID of the detection.
        - score (Union[float, int]): For ground truth, this is a flag
            indicating if the detection should be ignored (0) or included (1).
            For detection, this is the detection confidence score.
    """
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict: DefaultDict[
        int, List[Tuple[Tuple[float, ...], int, Union[float, int]]]
    ] = defaultdict(list)

    if not results_path.is_file():
        return results_dict

    with open(results_path, "r") as infile:
        for line in infile.readlines():
            # Don't have to rstrip() as float can handle newline character
            line_parts = line.split(",")
            # MOT GT format contains 9 elements, MOT DET format contains 7
            # elements
            if len(line_parts) < 7:
                continue
            entry = MOTEntry.create(line_parts, not is_gt and not is_ignore)
            # Frame numbers are 1-based, so this just ensures it
            if entry.frame_id < 1:
                continue

            if is_gt:
                if entry.score == 0 or entry.label not in valid_labels:
                    continue
            elif is_ignore:
                if entry.label not in ignore_labels and entry.vis_ratio >= 0:
                    continue
            results_dict[entry.frame_id].append(
                (entry.tlwh, entry.target_id, entry.score)
            )

    return results_dict
