**StratoLAMP – Duplex Quantification via Droplet Detection and Classification**

This project is tailored for StratoLAMP, a method enabling duplex nucleic acid quantification through the detection and classification of droplets. This classification is based on the stratification of precipitates within each droplet. The Mask R-CNN model (https://github.com/matterport/Mask_RCNN) has been customized to meet our specific requirements. We provide our modified Mask R-CNN model, which includes the training code, pre-trained weights, detection code, counting code, sample images, sample results, and other necessary files.

**Main Contents**
The repository comprises:
* Source code for constructing the Mask R-CNN model.
* Pre-trained weights for StratoLAMP.
* **train.py**: Training script for the droplet detection and classification model.
* **droplet_video_detect.py**: Detection script for performing droplet detection and classification on new images.
* **count_multi_types.py**: Counting script for obtaining the predicted numbers of four classes of droplets, essential for subsequent nucleic acid quantification.
* **Sample images/**: Folder containing sample images for testing and demonstration, including images of individual classes such as empty, low, medium, large droplets, and mixed classes.
* **results/**: Directory to store the results of droplet detection and classification. Prediction results for sample images using our model are included.

**Usage**
1. **Train Your Model**
We provide pre-trained weights for StratoLAMP, but you can use your dataset to train your own model. Follow these steps after preparing your dataset:
   * Adjust the **<N_train>** and **<N_val>** parameters in **droplet.py** based on the number of samples in your training and validation sets.
   * Run the **train.py** script to train a new model. Modify script parameters as needed.

2. **Detect Droplets in a New Image**
Use the **droplet_video_detect.py** script for droplet detection and classification on a new image.
   * Replace **<weight_path>** with the actual path to your adopted weight files.
   * Replace **<image_path>** with the actual path to your target image.
   * Replace **<save_path>** with the path where you want to save the results.
   * Run the **droplet_video_detect.py** script for detection.

3. **Counting the Number of Detected Images**
Use the **count_multi_types.py** script to obtain the predicted number of four classes in results.
   * Substitute **<json_path>** with the actual path to JSON files in your detection results.

**Examples**
Sample images in the **Sample images** folder can be used to test and demonstrate the detection and classification functionality. You can review the results in the **results** folder or run our model on your own.

**Notes**
* Ensure that project dependencies are installed (refer to requirements.txt in https://github.com/matterport/Mask_RCNN).
* Modify script paths and parameters to fit your project requirements.

Results
Detection and classification results will be saved in the **results/** folder.