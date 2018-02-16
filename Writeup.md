
# **Behavioral Cloning** 

## Writeup 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_summary.jpg "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_6_2.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results (this file)
* run.mp4 demonstrating model_6_2 doing a lap
* run_hidden.mp4 demonstrating hidden features at 2nd convolution layer for first 5 seconds of said lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_6_2.h5
#or
python drive.py model_0.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network taken directly from [this paper by NVIDIA].

[this paper by NVIDIA]: https://arxiv.org/pdf/1604.07316.pdf

The model includes RELU layers to introduce nonlinearity, and the data is normalized and cropped in the model using a Keras layers (code lines 103, 106 resp.). 

#### 2. Attempts to reduce overfitting in the model

By far the most important factor was 'incentive alignment'; just using the raw data and minimising MSE would incentivise the model to steer straight almost all the time because of the huge imbalance in the data. There were multiple options; instead of collecting fresh data only around curves or adding shear to all images with adjusted steering angle, I dropped 70% of low-steering angle data when collecting data, and augmented the remaining data with reflections. The 50% was chosen experimentally; too high and the automated model drove too wildly, too low and it didn't adjust fast enough to getting close to the edge of the road. What counted as 'low-steering' was also found experimentally; in the end I used 0.5 (around line 17).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 41).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in both senses around the track and smooth corners.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to stand on the shoulders of giants (see above). NVIDIA had already solved this problem (only with real data not simulations) and I wanted to take advantage of their considerable time investment in parameter tuning.

It turned out that this architecture was enough for my model to succeed - this is why I included two models above. Nonetheless I am using this project as an opportunity to practise the science of improving a model, so I spent time tweaking.

Initially my changes resulted in worse performance. As I added data, I expected limiting accuracy to improve - and it did. As I added more subtle data - the recovery behaviour - the accuracy dropped, which was expected. This represented a decrease in the proportion of images that were straight driving, not a worseneing of the model; in fact as it adapted to a more varied data set it did become a better driver despite a falling accuracy rate.

Some of the later changes to the model included transforming input images to YUV scale - I think this saved the network the task of finding one linear feature of the RGB images. It all helps!

Finally, using the near-straight dropping made the models work.

In the end the model_6 line work at least as well as model_0 on track 1, and there is some evidence they may generalise better on track 2 - though this track is too different to be hugely useful.

#### 2. Final Model Architecture

The final model architecture (model.py lines 98-140) consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric):

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from minor errors. Recovering from major errors is beyond the scope of the project since it would entail recognising semantically rich features like 'road', 'on road' and 'off road'. 

To augment the raw data sat, I flipped images and angles (reflected in vertical line).

After the collection process, I had 12000-23000 data points (12k for model_0, 23k for models that failed, then 17k for model_6; I deleted data I had recorded which proved too taxing for the model, or if I'm honest, data in which my driving was too bad). I then preprocessed this data by converting to YUV scale and cropping to remove the bonnet and sky. The bonnet was minor a waste of computing resources, while the sky was a waste of neural activity since the model would. at least initially, detect entirely contingent features such as "whenever two poplars line up next to a palm tree in the shade I must turn right". 

I finally randomly shuffled the data set and put 10/20% of the data into a validation set (model_6/model_0). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 to 5, as evidenced by the validation error troughing. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is an excerpt from my experimental log. Note that the increase in validation error is sometimes a sign of a better training set and so a more robust model.

    1. Smaller data set gives 1.45%, 1.62% on 5 epochs
    2. Adding road-corrections and a bit of bad driving gave 0.67% and 1.62% on 15 epochs!
    3. Removing training data of Feb 13th... 1 epoch => 1.08%, 0.94%. My driving is terrible!
    4. YUV, 5 epochs gets valid down to 0.92% (still above 0.86% of model_0...)
    5. Added more data including corrections, 0.94% & 1.09%
    6. Removing straight steering: 0.95%, 1.18% - but this is quite good considering the easy target is removed
    7. Removing 70% of straight steering: 0.94%, 0.83% (82 after epoch 4!) - model_6_1
    8. Removing 70% of straightISH steering - model_6_2



```python

```
