"""Evaluation helpers for mean average precision metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

from utils import bbox_utils


def init_stats(labels: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    """Create empty bookkeeping structures for each foreground label.

    Args:
        labels (Sequence[str]): Dataset labels including background at index ``0``.

    Returns:
        Dict[int, Dict[str, Any]]: Per-class metric accumulators keyed by label id.
    """
    stats = {}
    for index, label in enumerate(labels):
        if index == 0:
            continue
        stats[index] = {
            "label": label,
            "total": 0,
            "tp": [],
            "fp": [],
            "scores": [],
        }
    return stats


def update_stats(
    pred_bboxes: Any,
    pred_labels: Any,
    pred_scores: Any,
    gt_boxes: Any,
    gt_labels: Any,
    stats: Dict[int, Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    """Update per-class precision/recall bookkeeping for one batch.

    Args:
        pred_bboxes (Any): Predicted boxes for the batch.
        pred_labels (Any): Predicted class ids for the batch.
        pred_scores (Any): Predicted scores for the batch.
        gt_boxes (Any): Ground-truth boxes for the batch.
        gt_labels (Any): Ground-truth labels for the batch.
        stats (Dict[int, Dict[str, Any]]): Running metric state.

    Returns:
        Dict[int, Dict[str, Any]]: Updated metric state.
    """
    iou_map = bbox_utils.generate_iou_map(pred_bboxes, gt_boxes)
    merged_iou_map = tf.reduce_max(iou_map, axis=-1)
    max_indices_each_gt = tf.argmax(iou_map, axis=-1, output_type=tf.int32)
    sorted_ids = tf.argsort(merged_iou_map, direction="DESCENDING")

    # Count how many ground-truth instances exist for each class in the batch.
    count_holder = tf.unique_with_counts(tf.reshape(gt_labels, (-1,)))
    for index, gt_label in enumerate(count_holder[0]):
        if gt_label == -1:
            continue
        label_id = int(gt_label)
        stats[label_id]["total"] += int(count_holder[2][index])

    for batch_id, _ in enumerate(merged_iou_map):
        # A ground-truth box may only be matched to one prediction when computing AP.
        true_labels = []
        for sorted_id in sorted_ids[batch_id]:
            pred_label = pred_labels[batch_id, sorted_id]
            if pred_label == 0:
                continue

            iou = merged_iou_map[batch_id, sorted_id]
            gt_id = max_indices_each_gt[batch_id, sorted_id]
            gt_label = int(gt_labels[batch_id, gt_id])
            label_id = int(pred_label)
            score = pred_scores[batch_id, sorted_id]
            stats[label_id]["scores"].append(score)
            stats[label_id]["tp"].append(0)
            stats[label_id]["fp"].append(0)
            if iou >= 0.5 and label_id == gt_label and gt_id not in true_labels:
                stats[label_id]["tp"][-1] = 1
                true_labels.append(gt_id)
            else:
                stats[label_id]["fp"][-1] = 1
    return stats


def calculate_ap(recall: Any, precision: Any) -> float:
    """Compute the VOC-style 11-point interpolated average precision.

    Args:
        recall (Any): Recall curve values.
        precision (Any): Precision curve values aligned with ``recall``.

    Returns:
        float: Average precision for one class.
    """
    ap = 0.0
    for recall_threshold in np.arange(0, 1.1, 0.1):
        prec_rec = precision[recall >= recall_threshold]
        if len(prec_rec) > 0:
            ap += np.amax(prec_rec)
    return ap / 11


def calculate_mAP(stats: Dict[int, Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], float]:
    """Compute per-class AP values and their mean.

    Args:
        stats (Dict[int, Dict[str, Any]]): Per-class metric accumulators.

    Returns:
        Tuple[Dict[int, Dict[str, Any]], float]: Updated stats and the mean AP value.
    """
    aps = []
    for label in stats:
        label_stats = stats[label]
        tp = np.array(label_stats["tp"])
        fp = np.array(label_stats["fp"])
        scores = np.array(label_stats["scores"])
        # Sort detections by descending confidence before accumulating TP and FP counts.
        ids = np.argsort(-scores)
        total = label_stats["total"]
        accumulated_tp = np.cumsum(tp[ids])
        accumulated_fp = np.cumsum(fp[ids])
        recall = accumulated_tp / total
        precision = accumulated_tp / (accumulated_fp + accumulated_tp)
        ap = calculate_ap(recall, precision)
        stats[label]["recall"] = recall
        stats[label]["precision"] = precision
        stats[label]["AP"] = ap
        aps.append(ap)
    m_ap = np.mean(aps)
    return stats, m_ap


def evaluate_predictions(
    dataset: Any,
    pred_bboxes: Any,
    pred_labels: Any,
    pred_scores: Any,
    labels: Sequence[str],
    batch_size: int
) -> Dict[int, Dict[str, Any]]:
    """Evaluate an SSD prediction run and print the resulting mAP.

    Args:
        dataset (Any): Batched evaluation dataset.
        pred_bboxes (Any): Predicted boxes for the full evaluation split.
        pred_labels (Any): Predicted class ids for the full evaluation split.
        pred_scores (Any): Predicted scores for the full evaluation split.
        labels (Sequence[str]): Dataset labels including background.
        batch_size (int): Prediction batch size.

    Returns:
        Dict[int, Dict[str, Any]]: Final per-class evaluation statistics.
    """
    stats = init_stats(labels)
    for batch_id, image_data in enumerate(dataset):
        _, gt_boxes, gt_labels = image_data
        start = batch_id * batch_size
        end = start + batch_size
        batch_bboxes = pred_bboxes[start:end]
        batch_labels = pred_labels[start:end]
        batch_scores = pred_scores[start:end]
        stats = update_stats(batch_bboxes, batch_labels, batch_scores, gt_boxes, gt_labels, stats)
    stats, m_ap = calculate_mAP(stats)
    print("mAP: {}".format(float(m_ap)))
    return stats
