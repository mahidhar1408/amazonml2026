from typing import Dict, Set, Tuple

def compute_entity_er_metrics(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    candidate_sets: Dict[str, Set[str]] = None
) -> Dict[str, float]:
    """Compute per-Source 1 macro F_0.5 and auxiliary metrics.
    
    ground_truth: dict mapping s1_id -> set of true matched target IDs (empty set for singletons)
    predictions: dict mapping s1_id -> set of predicted matched target IDs
    candidate_sets: optional dict mapping s1_id -> set of candidate target IDs
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {}
        
    f05_scores = []
    precision_scores = []
    recall_scores = []
    
    singleton_total = 0
    singleton_correct = 0
    non_singleton_total = 0
    non_singleton_f05 = []
    
    candidate_hits = 0
    total_true_targets = 0
    
    for s1_id, true_targets in ground_truth.items():
        pred_targets = predictions.get(s1_id, set())
        
        # Candidate recall tracking if candidates provided
        if candidate_sets is not None:
            cands = candidate_sets.get(s1_id, set())
            total_true_targets += len(true_targets)
            candidate_hits += len(true_targets.intersection(cands))
            
        if len(true_targets) == 0:
            singleton_total += 1
            if len(pred_targets) == 0:
                singleton_correct += 1
                f05_scores.append(1.0)
                precision_scores.append(1.0)
                recall_scores.append(1.0)
            else:
                f05_scores.append(0.0)
                precision_scores.append(0.0)
                recall_scores.append(1.0)
        else:
            non_singleton_total += 1
            if len(pred_targets) == 0:
                f05_scores.append(0.0)
                precision_scores.append(1.0)  # No false positive merges
                recall_scores.append(0.0)
                non_singleton_f05.append(0.0)
            else:
                tp = len(true_targets.intersection(pred_targets))
                p = tp / len(pred_targets)
                r = tp / len(true_targets)
                precision_scores.append(p)
                recall_scores.append(r)
                if tp == 0:
                    f05 = 0.0
                else:
                    f05 = (1.25 * p * r) / (0.25 * p + r)
                f05_scores.append(f05)
                non_singleton_f05.append(f05)
                
    macro_f05 = sum(f05_scores) / total_entities
    macro_precision = sum(precision_scores) / total_entities
    macro_recall = sum(recall_scores) / total_entities
    
    res = {
        "macro_f05": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "singleton_accuracy": (singleton_correct / singleton_total) if singleton_total > 0 else 1.0,
        "singleton_count": singleton_total,
        "non_singleton_f05": (sum(non_singleton_f05) / len(non_singleton_f05)) if non_singleton_f05 else 0.0,
        "non_singleton_count": non_singleton_total,
        "total_entities": total_entities,
    }
    
    if candidate_sets is not None and total_true_targets > 0:
        res["candidate_recall"] = candidate_hits / total_true_targets
        total_candidates = sum(len(cands) for cands in candidate_sets.values())
        res["avg_candidates_per_s1"] = total_candidates / total_entities
        
    return res
