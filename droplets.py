import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from glob import glob
import io
import PIL.Image
import base64


def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    img_pil = PIL.Image.open(f).convert('RGB') # add
    return img_pil


def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr


def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr


def str_to_image(s):
    return img_b64_to_arr(s)

# 当前工作路径设置为往上两级父级目录，即 Mask_RCNN路径下
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model
from mrcnn import utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")  # 预训练模型的路径

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")  # 训练日志路径

# you need to change the number based on your training set and validation set
N_TRAIN = 70 #  训练集数据量，需修改
N_VAL = 6  #    验证集数据量，需修改


# 训练液滴检测网络时的设置
class DropletConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "droplet"
    BACKBONE = 'resnet50'  # 选择 resnet50 作为backbone
    GPU_COUNT = 3  # GPU数量

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + droplet types

    # Number of training steps per epoch
    STEPS_PER_EPOCH = (N_TRAIN + IMAGES_PER_GPU - 1) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, (N_VAL + IMAGES_PER_GPU - 1) // IMAGES_PER_GPU)

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0

    MEAN_PIXEL = np.array([239.11664895, 239.11664895, 239.11664895])

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.95
    PRE_NMS_LIMIT = 9000
    # Length of square anchor side in pixels

    # Input image resizing
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 768
    IMAGE_MIN_SCALE = 0

    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # BACKBONE_STRIDES = [2, 4, 8, 16, 16]
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 5000

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 1000

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 6000
    POST_NMS_ROIS_INFERENCE = 4000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    # MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    LEARNING_RATE = 0.005


class DropletInferenceConfig(DropletConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    # 这里要注意，缩放模式要跟训练时一致， 本实验中训练使用的为‘square’
    # IMAGE_RESIZE_MODE = "pad64"
    IMAGE_RESIZE_MODE = "square"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


#    IMAGE_RESIZE_MODE = "crop"
#    IMAGE_MIN_DIM = 512
#    IMAGE_MAX_DIM = 512
#    IMAGE_MIN_SCALE = 2.0

#    USE_MINI_MASK = False


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class DropletDataset(utils.Dataset):

    # 从特定路径下每张图像的json文件中读取类别信息和图像信息
    def load_droplet(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. Now, We have four classes to add.
        self.add_class("droplet", 1, "low_positive")
        self.add_class("droplet", 2, "negative")
        self.add_class("droplet", 3, "medium_positive")
        self.add_class("droplet", 4, "high_positive")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.

        # 返回 包含目标路径下所有json文件路径名的列表
        files = glob(os.path.join(dataset_dir, "*.json"))

        def _get_special_name(fname):
            return fname.rsplit('/', 1)[1].split('.')[0]

        # Add images
        for fname in files:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            with open(fname, 'r') as f:
                fcontent = json.load(f)
            height = fcontent['imageHeight']
            width = fcontent['imageWidth']
            self.add_image(
                "droplet",
                image_id=_get_special_name(fname),  # use file name as a unique image id
                path=fcontent['imagePath'],
                width=width, height=height,
                shapes=fcontent['shapes'],
                image=str_to_image(fcontent['imageData']))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.

        image_info = self.image_info[image_id]
        if image_info["source"] != "droplet":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        # 创建n张与图像大小相同的全为0（黑色）的二维数组， n----取决于被标记的区域的个数（即需要的mask个数，即len(info["shapes"])）
        mask = np.zeros([info["height"], info["width"], len(info["shapes"])],
                        dtype=np.uint8)

        class_ids = []

        for i, p in enumerate(info["shapes"]):
            # Get indexes of pixels inside the polygon and set them to 1
            points = np.array(p['points'])  # 单张图像中其中的一个标记区域的边的点集合
            rr, cc = skimage.draw.polygon(points[:, 1], points[:, 0])  # [rr, cc]表示依据标记的点集合使用skimage库函数绘制的一个多边形
            mask[rr, cc, i] = 1  # 将该图像中的标记区域（即绘制的多边形）赋值为 1（白色）
            # 需要修改， 此时类别数发生了变换， 需要将不同类别与类别id(数值型)一一对应上
            if p['label'] == 'low_positive':
                class_id = 1
            if p['label'] == 'medium_positive':
                class_id = 3
            if p['label'] == 'high_positive':
                class_id = 4
            if p['label'] == 'negative':
                class_id = 2
            class_ids.append(class_id)

        # 返回 mask (numpy数组, shape: [图像宽， 图像高， 实例数]) 和 class_ids (numpy数组， shape: [1, 实例数])
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def load_image(self, image_id):
        return self.image_info[image_id]['image']

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "droplet":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.

        image_info = self.image_info[image_id]
        if image_info["source"] != "precipitate":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        # 创建n张与图像大小相同的全为0（黑色）的二维数组， n----取决于被标记的区域的个数（即需要的mask个数，即len(info["shapes"])）
        mask = np.zeros([info["height"], info["width"], len(info["shapes"])],
                        dtype=np.uint8)

        class_ids = []
        global class_id

        for i, p in enumerate(info["shapes"]):
            # Get indexes of pixels inside the polygon and set them to 1
            points = np.array(p['points'])  # 单张图像中其中的一个标记区域的边的点集合
            rr, cc = skimage.draw.polygon(points[:, 1], points[:, 0])  # [rr, cc]表示依据标记的点集合使用skimage库函数绘制的一个多边形
            mask[rr, cc, i] = 1  # 将该图像中的标记区域（即绘制的多边形）赋值为 1（白色）
            # 需要修改， 此时类别数发生了变换， 需要将不同类别与类别id(数值型)一一对应上

            if p['label'] == 'low_positive':
                class_id = 1
            if p['label'] == 'medium_positive':
                class_id = 2
            if p['label'] == 'high_positive':
                class_id = 3
            class_ids.append(class_id)

        # 返回 mask (numpy数组, shape: [图像宽， 图像高， 实例数]) 和 class_ids (numpy数组， shape: [1, 实例数])
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def load_image(self, image_id):
        return self.image_info[image_id]['image']

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "precipitate":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
