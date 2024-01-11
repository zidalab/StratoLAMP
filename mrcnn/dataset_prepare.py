import os
import cv2
import json
import nrrd
import base64
import pydicom
import nibabel as nib
import numpy as np
from labelme import utils


def main():
    train_path = '/home/zy/Droplets_multi-types/train/'
    data_path = ''
    mask_path = ''

    positive_list, negative_list = classify_data_list(train_path)
    positive_image, positive_mask = format_data(positive_list)
    negative_image, negative_mask = format_data(negative_list)

    #print(positive_image, positive_mask, negative_image, negative_mask)
    data_total = get_data_total([], positive_image, positive_mask)
    data_total = get_data_total(data_total, negative_image, negative_mask)
    ''' 
    data_total: [{'id': identity of a patient, 
                 'image_path': path of the patient's image, 
                 'mask_path': path of the patient's mask}, ...]
    '''
    get_json_total(data_total)


def dict_json(imageData, shapes, imagePath, width, height):
    '''
    :param imageData: str
    :param shapes: list
    :param imagePath: str
    :param fillColor: list
    :param lineColor: list
    :return: dict
    '''
    return {"version": "4.5.7", "flags": {}, "shapes": shapes, 'imagePath': imagePath,
            "imageData": imageData, "imageHeight": width, "imageWidth": height}


def dict_shapes(label, points):
    return {"label": label, "points": points, "shape_type": "polygon", "flags": {}}


def get_json_total(data_total):
    for data in data_total:
        image_json_shapes = []  # 该帧图像json文件的'shapes'列表
        points = []
        if '.nrrd' in data['image_path']:
            image_Data, image_info = nrrd.read(data['image_path'])
        elif '.dcm' in data['image_path']:
            image_Data = pydicom.dcmread(data['image_path']).pixel_array


        mask = nib.load(data['mask_path']).get_data()
        mask = mask.astype(np.uint8).squeeze()
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        width, height = mask.shape[0], mask.shape[1]

        if len(image_Data.shape) == 3:
            image_Data = image_Data.transpose(1, 0, 2)
            #image_Data = cv2.cvtColor(image_Data, cv2.COLOR_RGB2GRAY)
        else:
            #image_Data = image_Data.transpose(1, 0)
            image_Data = image_Data*255
            image_Data = image_Data.astype(np.uint8)
            image_Data = cv2.cvtColor(image_Data, cv2.COLOR_GRAY2BGR)
        print(image_Data.shape)

        img_b64 = utils.img_arr_to_b64(image_Data).decode('utf-8')
        #img_b64 = base64.b64encode(image_Data)
        #img_b64 = bytes.decode(img_b64)

        mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for mask_contour in mask_contours:
            mask_contour = mask_contour.flatten()
            mask_contour = mask_contour.tolist()
            for index in range(len(mask_contour)):
                if index % 2 == 0:
                    points.append([mask_contour[index], mask_contour[index + 1]])
        class_id = 0 if 'L' in data['image_path'] else 1     # 类别判断  阳性为1， 阴性为0
        image_json_shapes.append(dict_shapes('negative', points)) if class_id == 0 else image_json_shapes.append(dict_shapes('positive', points))
        image_data_json = dict_json(img_b64, image_json_shapes, data['image_path'], width, height)
        json_save_path = '/home/zy/Droplets_multi-types/PHPT_json/'
        json_save_name = json_save_path + str(data['id']) + '.json'  # json文件保存路径名
        json.dump(image_data_json, open(json_save_name, 'w'))  # 保存该帧检测结果的json文件
    return


def get_data_total(data_total, image, mask):
    for index in range(len(image)):
        assert (os.path.splitext(os.path.splitext(os.path.basename(mask[index]))[0])[0] == os.path.splitext(os.path.basename(image[index]))[0])
        data = {'id': os.path.splitext(os.path.basename(image[index]))[0],
                'image_path': image[index],
                'mask_path': mask[index]
                }
        data_total.append(data)
    return data_total


def classify_data_list(path):
    positive_list = []
    negative_list = []
    for dir in os.listdir(path):
        if 'H' in dir:
            positive_list.append(os.path.join(path, dir))
        elif 'L' in dir:
            negative_list.append(os.path.join(path, dir))
    return positive_list, negative_list


def format_data(path_list):
    mask_files_total = []
    image_files_total = []
    for dir in path_list:
        all_files = os.listdir(dir)
        mask_files = [os.path.join(dir, file) for file in all_files if '.nii' in file]
        image_files = [os.path.join(dir, file) for file in all_files if '.nrrd' in file or '.dcm' in file]
        mask_files_total.extend(mask_files)
        image_files_total.extend(image_files)
    mask_files_total.sort()
    image_files_total.sort()
    return image_files_total, mask_files_total


if __name__ == "__main__":
    main()
