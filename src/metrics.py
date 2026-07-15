import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    average_precision_score,
    classification_report
)


def evaluate_multilabel_metrics(y_true, y_pred_probs, label_map=None, threshold=0.35):
    """
    Calculates comprehensive multi-label classification metrics including Exact Match, Micro/Macro F1, mAP, and Class-wise breakdowns.
    :param y_true: ground truth binary matrix of shape [N, num_classes].
    :param y_pred_probs: predicted sigmoid probabilities matrix of shape [N, num_classes].
    :param label_map: dictionary mapping string labels to integer IDs.
    :param threshold: float confidence threshold to convert probabilities into binary predictions.
    :return: dictionary containing numerical metric summaries.
    """

    # Convert probability matrix to binary predictions using the custom threshold:
    y_pred_bin = (y_pred_probs >= threshold).astype(np.int32)
    y_true_bin = y_true.astype(np.int32)

    # Exact Match Ratio (Subset Accuracy - requires 100% perfect vector match):
    exact_match = accuracy_score(y_true_bin, y_pred_bin) * 100.0

    # Micro-Averaged Metrics (Global aggregate across all action instances):
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='micro', zero_division=0
    )

    # Macro-Averaged Metrics (Unweighted average across classes - reveals rare class performance):
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average='macro', zero_division=0
    )

    # Mean Average Precision (mAP - evaluates threshold-independent ranking confidence):
    try:
        map_score = average_precision_score(y_true_bin, y_pred_probs, average='macro') * 100.0
    except ValueError:
        map_score = 0.0

    # Print Clean Metric Summary Dashboard:
    print("\n" + "="*60)
    print("           FINAL MULTI-LABEL MODEL EVALUATION           ")
    print("="*60)
    print(f"Exact Match Ratio (Subset Acc) : {exact_match:.2f}%")
    print(f"Mean Average Precision (mAP)   : {map_score:.2f}%")
    print("-" * 60)
    print(f"Micro Precision : {micro_prec * 100:.2f}% | Micro Recall : {micro_rec * 100:.2f}% | Micro F1 : {micro_f1 * 100:.2f}%")
    print(f"Macro Precision : {macro_prec * 100:.2f}% | Macro Recall : {macro_rec * 100:.2f}% | Macro F1 : {macro_f1 * 100:.2f}%")
    print("="*60)

    # Per-Class Detailed Breakdown Report:

    if label_map:
        # Invert label map to get ordered class names:
        target_names = [name.upper() for name, idx in sorted(label_map.items(), key=lambda item: item[1])]
        print("\n--- Per-Class Performance Breakdown ---")
        print(classification_report(y_true_bin, y_pred_bin, target_names=target_names, zero_division=0))
        print("="*60 + "\n")

    return {
        "exact_match": exact_match,
        "map": map_score,
        "micro_f1": micro_f1 * 100.0,
        "macro_f1": macro_f1 * 100.0
    }
