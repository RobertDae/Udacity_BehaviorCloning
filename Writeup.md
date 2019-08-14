# Behavioral Cloning 

## Writeup

##### Author: Robert Dämbkes

The goals/steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) 
individually and describe how I addressed each point in my implementation.  

---
### Files Submitted and Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* [model.h5](./model.h5) containing a trained convolution neural network 
* [Writeup.md](./Writeup.md) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim) 
and my [drive.py](./drive.py) file, the car can be driven autonomously around 
the track by executing 
```sh
python drive.py model.h5
```

#### 3. Code is usable and readable

The [model.py](./model.py) file contains the code for training and 
saving the convolution neural network. The file [nn.py](./nn.py) shows the pipeline I used 
for training and validating the model, and it contains comments to explain how 
the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used neural network architecture similar to one, that has been proposed
by Nvidia in the 
[article](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).
[ModelDetails]./images/NVidia_Network_For_AD.JPG

I have slightly changed the model though. Cropping layer was added that transforms
the image of shape (160, 320) to (70, 320). There is also a normalization layer
(a Keras lambda layer). The model includes RELU layers to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

The model does not contain dropout layers. I just simply collected enough data 
to prevent overfitting. The data collection strategy is described later in this
writeup.

The model was trained and validated on different data sets to ensure 
that the model was not overfitting. The model was tested by running it 
through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving and recovering from the left and right sides 
of the road. Initially there were 3 center lane driving laps on track 1.
There were also 1 recovery laps per each track. Later I have added 1 center lane driving laps 
per each track and more recovery data for tricky places.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used Nvidia's model architecture because they have successfully applied it to the similar problem.
I've also tried [Comma.ai's model architecture](https://github.com/commaai/research/blob/master/train_steering_model.py)
however, it did not perform as good as Nvidia's.

At first, I have tried Nvidia's architecture, but with BGR image input. There was a problem with overall stability 
of the driving and with certain turns on the second half of the first track.

Comma.ai's model performed slightly better with the same input (BGR image), but it still had some 
problems with certain edges of the road on the first track. However, this was the first model that was able to autonomously drive 
the full lap on the track 1. Unfortunately, the driving behavior was not satisfactory.

Then I have collected more data and read the following articles:
- [Vivek Yadav's article](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)
- [Mengxi Wu's article](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234)
- [Denise R. James' article](https://medium.com/@deniserjames/denise-james-bsee-msee-5beb448cf184)

One idea also would be changing the image preprocessing using the S (saturation) channel from the HSV-encoded image. 
The contours of the roadway would be clearly visible in S channel.

I have slightly changed the number of neurons in dense layers in Nvidia's model. 
Then I started training this model again from scratch, and after only 2 epoch it was able to 
autonomously drive the car around the track 1 without even touching lane lines on the edges of the road.

I have collected in total 11616 images with corresponding steering angles and augmented this data by
flipping each image to prevent left or right biases in angles. I got 11616 images data set in total.
The data archive can be obtain using the [link](https://yadi.sk/d/PaOHVil33HnCKz).

The data set was split into training and validation data sets in a 4:1 proportion (i.e., 20% of images in the validation data set).
Images were also shuffled appropriately.

At the end of the process, the vehicle can drive autonomously around the track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a Nvidia-like convolution neural network with the following layers and layer sizes.
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 70, 320, 1)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 70, 320, 1)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 33, 158, 24)   624         lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4224)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           422500      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 30)            1530        dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             31          dense_3[0][0]                    
====================================================================================================
Total params: 598,259
Trainable params: 598,259
Non-trainable params: 0
____________________________________________________________________________________________________
```

##### 2.1 Output of the Keras model (while training):
```
Using TensorFlow backend.
Epoch 1/2
2019-08-11 20:21:36.037935: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2019-08-11 20:21:36.038007: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2019-08-11 20:21:36.038019: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2019-08-11 20:21:36.038051: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2019-08-11 20:21:36.038081: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2019-08-11 20:21:36.141511: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-08-11 20:21:36.142211: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties: 
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.09GiB
2019-08-11 20:21:36.142256: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0 
2019-08-11 20:21:36.142278: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y 
2019-08-11 20:21:36.142294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
6192/6192 [==============================] - 3826s 618ms/step - loss: 0.0033 - val_loss: 0.0253
Epoch 2/2
6192/6192 [==============================] - 3796s 613ms/step - loss: 4.0206e-04 - val_loss: 0.0258

````
[FinalVideoInAutonomousMode](./Videos/run1.mp4)

#### 3. Creation of the Training Set and Training Process

To capture good driving behavior, I recorded laps on tracks 1 using center lane driving. Then I recorded the vehicle 
recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from undesired situations.

To augment the data set, I also flipped images and angles thinking that this would mitigate the left angles biases in the dataset.

My functions to help the data augmetation like image flipping can be found in [utils.py](./utils.py).

Here is some pictures and graphs that helped me to understand the data.

###### Center, Left and Right Images Taken at the Same Moment in Time
![IMG](images/center_left_right_camera.png)

###### Flipped Center, Left and Right Images Taken at the Same Moment in Time
![IMG](images/center_left_right_camera_flipped.png)

###### Center, Left and Right Images as an Input to the First Convolutional Layer
![IMG](images/center_left_right_camera_conv_input.png)


I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used an adam optimizer so that manually training the learning rate was not necessary.
