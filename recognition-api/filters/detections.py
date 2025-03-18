import numpy as np
from scipy.spatial import distance

def filter_and_complete_detections(detections):
    """
    Processes YOLO detection results to remove false positives and 
    complete missing detections if one of the four expected objects is missing.
    
    Args:
        detections (list of tuples): Each tuple contains 
            (class, confidence, x, y, w, h)

    Returns:
        list of tuples: A list of the best 4 detections after filtering.
    """
    confidence_threshold = 0.5  # Set a confidence threshold to filter weak detections
    candidates = [d for d in detections if d[1] >= confidence_threshold]

    # If there are less than 3 detections, return as it is difficult to correct
    if len(candidates) < 3:
        return candidates  

    # Select the best 4 detections based on geometric consistency
    selected = select_best_4(candidates)

    # If less than 4 detections remain, try to complete the missing one
    if len(selected) < 4:
        selected = complete_missing_detection(selected, candidates)

    return selected

def select_best_4(candidates):
    """
    Selects the most reasonable set of 4 objects based on geometric consistency.

    Args:
        candidates (list of tuples): The filtered YOLO detections.

    Returns:
        list of tuples: The best 4 detections based on geometric error.
    """
    points = np.array([(d[2], d[3]) for d in candidates])
    best_set = []
    min_error = float('inf')

    # Try all combinations of 4 detections and find the best set
    from itertools import combinations
    for combo in combinations(range(len(candidates)), 4):
        selected_points = points[list(combo)]
        error = compute_geometric_error(selected_points)

        # Keep the combination with the smallest geometric error
        if error < min_error:
            min_error = error
            best_set = [candidates[i] for i in combo]

    return best_set

def compute_geometric_error(points):
    """
    Evaluates how well 4 points form a rectangular pattern.

    Args:
        points (numpy array): Array of 4 (x, y) coordinates.

    Returns:
        float: The geometric error, where lower values indicate better alignment.
    """
    if len(points) != 4:
        return float('inf')

    # Compute the pairwise distances between all points
    dist_matrix = distance.cdist(points, points)
    mean_dist = np.mean(dist_matrix)

    # Compute the sum of absolute differences from the mean distance
    error = np.sum(np.abs(dist_matrix - mean_dist))
    return error

def complete_missing_detection(selected, candidates):
    """
    Completes the missing detection if only 3 objects were found.

    Args:
        selected (list of tuples): The best detections selected so far.
        candidates (list of tuples): The original set of candidates.

    Returns:
        list of tuples: A corrected list of 4 detections.
    """
    if len(selected) == 4:
        return selected  # No need to complete if already 4

    selected_points = np.array([(d[2], d[3]) for d in selected])
    remaining_candidates = [d for d in candidates if d not in selected]

    best_candidate = None
    min_error = float('inf')

    # Find the best candidate to serve as the missing fourth point
    for candidate in remaining_candidates:
        new_set = np.vstack([selected_points, [candidate[2], candidate[3]]])
        error = compute_geometric_error(new_set)

        if error < min_error:
            min_error = error
            best_candidate = candidate

    # If a valid candidate is found, add it
    if best_candidate:
        selected.append(best_candidate)
    else:
        # If no suitable candidate exists, estimate the missing point
        selected.append(predict_missing_point(selected))

    return selected

def predict_missing_point(selected):
    """
    Predicts the missing fourth point based on geometric alignment.

    Args:
        selected (list of tuples): The current 3 selected detections.

    Returns:
        tuple: A pseudo-detection tuple (class, confidence, x, y, w, h).
    """
    p1, p2, p3 = np.array([(d[2], d[3]) for d in selected])
    
    # Estimate the fourth point based on the existing three
    p4_x = p1[0] + (p3[0] - p2[0])
    p4_y = p1[1] + (p3[1] - p2[1])

    # Estimate width and height based on the average of existing detections
    w_avg = np.mean([d[4] for d in selected])
    h_avg = np.mean([d[5] for d in selected])

    # Assign a moderate confidence score (e.g., 0.5)
    return (selected[0][0], 0.5, p4_x, p4_y, w_avg, h_avg)
