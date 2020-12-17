# Facial-Keypoint-Detection
----------------------------

---> Overview:
--------------------------------------------------------------------------------------------------------------------------------------------------
In this project, I combined my knowledge of computer vision techniques and deep learning architectures
to build a facial keypoint detection system that takes in any image with faces, and predicts the location 
of 68 distinguishing keypoints on each face.
--------------------------------------------------------------------------------------------------------------------------------------------------

---> The employed architecture:
--------------------------------------------------------------------------------------------------------------------------------------------------

I used "NaimishNet" which is proposed in "Facial Key Points Detection using Deep Convolutional 
Neural Network - NaimishNet" 

----------------------------------
NAIMISHNET LAYER-WISE ARCHITECTURE
----------------------------------
1- Input1
2- Convolution2d1 
3- Activation1 
4- Maxpooling2d1 
5- Dropout1 
6- Convolution2d2 
7- Activation2 
8- Maxpooling2d2 
9- Dropout2 
10- Convolution2d3 
11- Activation3 
12- Maxpooling2d3 
13- Dropout3 
14- Convolution2d4 
15- Activation4 
16- Maxpooling2d4 
17- Dropout4 
18- Flatten1 
19- Dense1 
20- Activation5
21- Dropout5
22- Dense2 
23- Activation6 
24- Dropout6 
25- Dense3 

• Input1 is the input layer.
• Activation1 to Activation5 use Exponential Linear Units (ELUs) as activation functions, whereas Activation6 uses Linear Activation Function.
• Dropout  probability is increased from 0.1 to 0.6 from Dropout1 to Dropout6, with a step size of 0.1.
• Maxpooling2d1 to Maxpooling2d4 use a pool shape of (2, 2), with non-overlapping strides and no zero padding.
• Flatten1 flattens 3d input to 1d output.
• Convolution2d1 to Convolution2d4 do not use zero padding, have their weights initialized with random numbers drawn from uniform distribution.
• Dense1 to Dense3 are regular fully connected layers with weights initialized using Glorot uniform initialization.
• Adam optimizer, with learning rate of 0.001, β1 of 0.9, β2 of 0.999 and ε of 1e−08, is used for minimizing Mean Squared Error (MSE).

  Layer Name          Number of Filters         Filter Shape
--------------------------------------------------------------
Convolution2d1            32                       (4, 4)
Convolution2d2            64                       (3, 3)
Convolution2d3           128                       (2, 2)
Convolution2d4           256                       (1, 1)
--------------------------------------------------------------------------------------------------------------------------------------------------

---> Before using the code
--------------------------------------------------------------------------------------------------------------------------------------------------
1- uzip images and detector_architectures folders
2- create a saved_models folder
