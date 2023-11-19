StratoLAMP – Duplex Quantification via Droplet Detection and Classification

This project is tailored for StratoLAMP, a method enabling duplex nucleic acid quantification through the detection and classification of droplets. This classification is based on the stratification of precipitates within each droplet. The Mask R-CNN model (https://github.com/matterport/Mask_RCNN) has been customized to meet our specific requirements. We provide our modified Mask R-CNN model, which includes the training code, pre-trained weights, detection code, counting code, sample images, sample results, and other necessary files.

Main Contents
The repository comprises:
•	Source code for constructing the Mask R-CNN model.
•	Pre-trained weights for StratoLAMP.
•	train.py: Training script for the droplet detection and classification model.
•	droplet_video_detect.py: Detection script for performing droplet detection and classification on new images.
•	count_multi_types.py: Counting script for obtaining the predicted numbers of four classes of droplets, essential for subsequent nucleic acid quantification.
•	Sample images/: Folder containing sample images for testing and demonstration, including images of individual classes such as empty, low, medium, large droplets, and mixed classes.
•	results/: Directory to store the results of droplet detection and classification. Prediction results for sample images using our model are included.

Usage
1.	Train Your Model
We provide pre-trained weights for StratoLAMP, but you can use your dataset to train your own model. Follow these steps after preparing your dataset:
•	Adjust the <N_train> and <N_val> parameters in droplet.py based on the number of samples in your training and validation sets.
•	Run the train.py script to train a new model. Modify script parameters as needed.
•	
2.	Detect Droplets in a New Image
Use the droplet_video_detect.py script for droplet detection and classification on a new image.
•	Replace <weight_path> with the actual path to your adopted weight fi