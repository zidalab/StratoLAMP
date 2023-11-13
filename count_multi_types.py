import json
import os


def count_all_frames(json_dir_path):
    json_files_name = [json_dir_path + x for x in os.listdir(json_dir_path)]
    json_files_name.sort()
    # Initialize
    weak_positive_count = strong_positive_count = weak_strong_mix_positive_count = count = 0
    for json_file_name in json_files_name:
        with open(json_file_name) as f:
            frame_specific = json.load(f)
        frame_rois = frame_specific['shapes']
        for roi in frame_rois:
            if roi['label'] == 'weak_positive':
                weak_positive_count += 1
            elif roi['label'] == 'strong_positive':
                strong_positive_count += 1
            elif roi['label'] == 'weak_strong_mix_positive':
                weak_strong_mix_positive_count += 1
            count += 1
    positive_detect_total = weak_positive_count + strong_positive_count + weak_strong_mix_positive_count
    accurate_weak = weak_positive_count / positive_detect_total
    accurate_strong = strong_positive_count / positive_detect_total
    accurate_weak_mix_strong = weak_strong_mix_positive_count / positive_detect_total
    return accurate_weak, accurate_strong, accurate_weak_mix_strong, count


def count_single_frame(json_path):
    # Initialize
    low_positive_count = high_positive_count = medium_positive_count = negative_count = count = 0
    with open(json_path) as f:
        frame_specific = json.load(f)
    frame_rois = frame_specific['shapes']
    for roi in frame_rois:
        if roi['label'] == 'low_positive':
            low_positive_count += 1
        elif roi['label'] == 'high_positive':
            high_positive_count += 1
        elif roi['label'] == 'medium_positive':
            medium_positive_count += 1
        elif roi['label'] == 'negative':
            negative_count += 1
        count += 1
    return count, negative_count, low_positive_count, medium_positive_count, high_positive_count


'''
主函数
'''
if __name__ == "__main__":
    json_path = './results/000Response/1018_poly/5/json'
    json_files = os.listdir(json_path)
    json_files.sort()
    negative_count_total = low_positive_count_total = medium_positive_count_total = high_positive_count_total = count = 0
    for i in range(len(json_files)):
        count_total, negative_count, low_positive_count, medium_positive_count, high_positive_count = \
            count_single_frame(os.path.join(json_path, json_files[i]))
        print("Path:", json_files[i])
        print("negative_count:", negative_count)
        print("low_positive_count:", low_positive_count)
        print("medium_positive_count:", medium_positive_count)
        print("high_positive_count", high_positive_count)
        print("count_total", count_total, '\n')
        negative_count_total += negative_count
        low_positive_count_total += low_positive_count
        medium_positive_count_total += medium_positive_count
        high_positive_count_total += high_positive_count
        count += count_total
    print("negative_count_total:", negative_count_total)
    print("low_positive_count_total:", low_positive_count_total)
    print("medium_positive_count_total:", medium_positive_count_total)
    print("high_positive_count_total", high_positive_count_total)
    print("count_total", count, '\n')


