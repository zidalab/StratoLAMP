"""
object counting
"""
from collections import defaultdict

def parse_detect_results(result):
    rois = result['rois']  # [n, 4]
    class_ids = result['class_ids']  # [n]
    masks = result['masks']  # [h, w, n]
    area = masks.sum(axis=[0, 1])
    count = defaultlist(int)
    area_dict = defaultlist(list)
    for i, a in zip(class_ids, area):
        count[i] += 1
        area_dict[i].append(a)
    return count, area_dict
