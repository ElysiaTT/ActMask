"""Dependency-light K-way retrieval metrics shared by RM-OD audits."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def group_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    recall1: list[float] = []
    recall3: list[float] = []
    reciprocal: list[float] = []
    ndcg: list[float] = []
    pair_order: list[float] = []
    count = 0
    for row in rows:
        scores = np.asarray(row["scores"], dtype=np.float64)
        positive = int(row["positive_index"])
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        rank = order.index(positive) + 1
        recall1.append(float(rank == 1))
        recall3.append(float(rank <= 3))
        reciprocal.append(1.0 / rank)
        ndcg.append(1.0 / np.log2(rank + 1))
        for index, score in enumerate(scores):
            if index != positive:
                pair_order.append(float(scores[positive] > score) + 0.5 * float(scores[positive] == score))
        count += 1
    return {"recall_at_1": float(np.mean(recall1)), "recall_at_3": float(np.mean(recall3)), "mrr": float(np.mean(reciprocal)), "ndcg": float(np.mean(ndcg)), "pair_order_accuracy": float(np.mean(pair_order)), "groups": count}


def score_rows(groups: Iterable[dict[str, Any]], scorer: Any) -> list[dict[str, Any]]:
    return [{"group_id": group["group_id"], "positive_index": group["positive_index"], "scores": [float(value) for value in scorer(group)]} for group in groups]
