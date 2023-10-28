# NVE-DGCNN (Normal Vector Enhanced Dynamic Graph Convolutional Neural Network) for 3D Surface Defect Segmentation
![Alt Text](https://github.com/fardinbh/NVE-DGCNN/blob/main/images/model.jpg?raw=true)
---

## Overview

This repository provides an implementation of the Normal Vector Enhanced DGCNN (NVE-DGCNN), an adapted version of the DGCNN, for semantic segmentation of concrete surface defects, primarily focusing on spalls and cracks. This method is a part of the research conducted by Fardin Bahreini, Ph.D., at Concordia University under the supervision of Dr. Amin Hammad.

Incorporated in this repo is a 3D point cloud dataset curated at Dr. Amin Hammad's lab, featuring 102 segments from four reinforced concrete bridges in Montreal, scanned using the FARO Focus3D scanner.

**Environment:**
- TensorFlow-GPU: 1.15.1
- CUDA: 11.0
- Python: 3.6

---

## Data Preparation

1. **Elimination and Registration**: After the data collection, the irrelevant points are removed, and the scans are prepped for registration. 
   
2. **Segmentation and Annotation**: Areas are extracted from the registered point cloud data, and respective parts within these areas are segmented. Manual annotations are performed on these parts, classifying them into three categories: crack, spalling, and non-defect.

3. **Augmentation**: To enhance the dataset, point clouds are flipped along the YZ plane.

![Alt Text](https://github.com/fardinbh/NVE-DGCNN/blob/main/images/Annotation.jpg?raw=true)
---

## Data Pre-processing

Two distinct approaches have been employed to prepare the dataset for the CNN's MLP classifier:

1. **XYZRGBL Conversion**: The original dataset files are transformed into data label files, presenting 2D matrices with XYZRGBL on each line. Each of these parts is then split into blocks, with normalized Y surface location values added. Each point gets represented as a 7-dimensional vector.

2. **Normal Vector Addition**: A hand-crafted point feature, the normal vector (Nx, Ny, Nz), is integrated for the NVE-DGCNN's MLP classifier. Research has shown that including these normal vectors can potentially enhance the performance of CNN networks in semantic segmentation tasks.

---

## Training and Evaluation

The segmentation model of DGCNN comprises a sequence of three EdgeConv layers followed by three fully-connected ones. The K-nearest neighbors count for EdgeConv layers, set in accordance with Wang et al.'s recommendation, is 20.

---

## Testing

To validate the model's accuracy, unseen dataset portions, exempt from the training and evaluation phases, are used. Performance metrics such as the confusion matrix, overall accuracy, recall, precision, F1 score, and Intersection over Union (IoU) are utilized to assess the model's segmentation capabilities. Emphasis is placed on recall given the critical nature of detecting actual defect points in concrete surface inspection.
NVE-DGCNN resulted in `98.56%` and `96.50%` recalls for semantic segmentation of cracks and spalls, respectively.

![Alt Text](https://github.com/fardinbh/NVE-DGCNN/blob/main/images/Test.png?raw=true)
---

## Publication

Bahreini, F. and Hammad, A., “Dynamic Graph CNN Based Semantic Segmentation of Concrete Defects and As-inspected Modeling” Journal of
Automation in Construction (Under Review 2023)

Bahreini, F. and Hammad, A., “Point Cloud Semantic Segmentation of Concrete Surface Defects Using Dynamic Graph CNN” In Proceedings of the 38th ISARC,
Dubai, UAE (2021)

