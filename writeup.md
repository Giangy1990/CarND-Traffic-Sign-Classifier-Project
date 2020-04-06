# **Traffic Sign Recognition**
---
This writeup describes the content of the traffic sign classifier project. The purpose is to design a deep convolution neural network able to classify a set of traffic signs. The network is trained and evaluated over the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)\
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./_Writeup-Imgs_/signs_distribition.PNG "Class distribution over all datasets"
[image2]: ./_Writeup-Imgs_/train_set_distribution.png "Classes distribution over train set"
[image3]: ./_Writeup-Imgs_/normalization.png "Normalization"
[image4]: ./_Writeup-Imgs_/preprocessing.png "Preprocessing pipeline"
[image5]: ./_Writeup-Imgs_/lenet_arch.jpg "LeNet"
[image6]: ./_Writeup-Imgs_/net_eval.png "Train Results"
[image7]: ./_Writeup-Imgs_/new_imgs.png "New Images"
[image8]: ./_Writeup-Imgs_/new_img_eval.png "New Images Classification"
[image9]: ./_Writeup-Imgs_/CNN_in.png "CNN input"
[image10]: ./_Writeup-Imgs_/CNN_inner.png "Feature Map"
---
## Data Set Summary & Exploration

### 1.Basic summary.

In **cell 2**, I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ***34799***
* The size of the validation set is ***4410***
* The size of test set is ***12630***
* The shape of a traffic sign image is ***(32, 32, 3)***
* The number of unique classes/labels in the data set is ***43***


### 2. Visualization of the dataset.

Here is an exploratory visualization of the data set.
In **cell 3** there are shown 43 pairs of image, one for each class contained in the dataset. The pair contains a pie that represents the distribution of the current class over the three datasets, and the image of the current class.\
Below there is an example:

![alt text][image1]

In **cell 4** there are shown three bar plots, for Training Set, Validation Set and Test Set, respectively.\
Below, the bar plot of the training set is showed as example:

![alt text][image2]

From this image it is possible to see that the distribution of the classes over the training set is unbalanced.\
N.B. The x-axis represents the percentage over the entire dataset.\
For example, take a look to the first two classes:
- *Speed limit (20km/h)*: **0.517 %**
- *Speed limit (30km/h)*: **5.690 %**

## Design and Test a Model Architecture

### 1. Pre-processing

Here is described the pre-processing pipeline useful to improve the performances of the neural network.\
This process is split into three phases:
- ***Shuffling***: it is useful to remove dependancies on set ordering; (**cell 5**)
- ***Gray scale conversion***: it is performed since the color information is not useful to the current classification and also because the LetNet was traditionally defined to work with gray scale image; (**cell 6**)
- ***Min-Max scaling***: it is useful to improve the gradient descend performance removing mean and normalizing the input value of the network; (**cell 6**)
The image below provides a clear view of the normalization effect on the gradient descend algorithm.

![alt text][image3]

Here is an example of a traffic sign image before and after applying the pre-processing pipeline.

![alt text][image4]

### 2. Model Architecture
The chosen neural network architecture is the [LeNet-5](http://yann.lecun.com/exdb/lenet/). It's logical representation is showed below

![alt text][image5]

* ***Input*** The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels.
Since the current dataset has been preprocessed in grayscale, C is 1.
* ***Output*** Return the class extracted by last fully connected layer.

My final model consisted of the following layers:

| Layer         		|     Description       		 | Output       |
|:-----------------:|:--------------------------:|:------------:|
| Convolution 5x5   | 1x1 stride, valid padding  | 28x28x6    	|
| RELU					    |	Rectified Linear Units     | 28x28x6    	|
| Max pooling	      | 2x2 stride, valid padding  | 14x14x6 			|
| Convolution 5x5   | 1x1 stride, valid padding  | 10x10x16    	|
| RELU					    |	Rectified Linear Units     | 10x10x16    	|
| Max pooling	      | 2x2 stride, valid padding  | 5x5x16   		|
| Flatten	          |                            | 400          |
| Dropout	          | keep probability 0.5       | 400          |
| Fully connected		|                            | 120          |
| RELU					    |	Rectified Linear Units     | 120        	|
| Fully connected		|                            | 84           |
| RELU					    |	Rectified Linear Units     | 84         	|
| Fully connected		|                            | 43           |

The main difference of this network with respect to the original one, is the introdction of a **Dropout** layer.\
This layer introduces redundant representation between the nodes to make the full network more robust.
The code of the network is contained in the **cell 8**.

### 3. Hyperparameters and optimization methods.
In the **cell 9** and **cell 10** are reported the hyperparameters and the optimization method used during the training phase.
In particular, I used the following hyperparameters:
* EPOCHS = 65
* BATCH_SIZE = 1024
* rate = 0.001

and the following operators:
* cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
* loss_operation = tf.reduce_mean(cross_entropy)
* optimizer = tf.train.AdamOptimizer(learning_rate = rate)
* training_operation = optimizer.minimize(loss_operation)

### 4. Training.
During the training procedure, I tuned two hyperparameters, EPOCHS and BATCH_SIZE.\
I started with EPOCHS = 20 and BATCH_SIZE = 512 to have a first accuracy reference for the next tuning.\
After some attemts, I decided to use the following values:
* EPOCHS = 65
* BATCH_SIZE = 1024

reaching the following accuracy over the three dataset:
* training set accuracy of 0.9964
* validation set accuracy of 0.9433
* test set accuracy of 0.9374

In the following image are showed the train results:

![alt text][image6]

Starting from left:
* accuracy evaluation over the training epochs using validation datasets;
* loss function behavior over the training epochs;
* final accuracies over the three datasets.

## Test a Model on New Images

### 1. Choose different German traffic signs.

Here are ten German traffic signs that I found on the web:

![alt text][image7]

These images might be difficult to classify because they have different light conditions and orientation with respect to the ones present in the training dataset.

### 2. Classification evaluation.
Running the trained network over these new images, I obtained an accuracy of 0.4, that is pretty low.
In the following image are shown the classification results

![alt text][image8]

The model was able to correctly guess 4 of the 10 traffic signs, which gives an accuracy of 40%.
The reason of the misclassification of the **Bumpy road** and **Speed limit (30km/h)** could be the unusual orientation of the sign. Data agumentation can help to reduce this type of misclassification.\
The misclassification of the **No passing** sign could be found in the horizontal compression of the image due to the resize performed in the preprocessing phase. Indeed, looking at the distortion of the image final image, it is very hard to recognize the original sign. This can be solved performing a better resize algorithm that doesn't introduce image distortion.\
The misclassification of the **Turn left head** and the **Roundabout mandatory** sign could be due to the different type of arrow and orientation of the sign. This can be solved increasing the training dataset and agumentation.\

### 3. Conclusion
The model was able to correctly guess 4 of the 10 traffic signs, which gives an accuracy of 40%.

These under light the need of robustness of the network with respect to orientation and more in general on different conditions of traffic signs.

## (Optional) Visualizing the Neural Network
The image below shows the first CNN layer input and respective feature maps:
* Input\
![alt text][image9]
* Output\
![alt text][image10]
From this image it is clear that this layer is selecting edges.

## Known Limitations and Open Points

Some known limitations of the neural networks are:
* Learning on Unbalanced Dataset
* Robustness over orientation

An open point that can be addressed is the data augmentation balanced dataset.
