# ChartImage-Classification-using-CNN
## Problem Statement:
- classifying a given chart image to one of five chart classes, namely “Line”, “Dot Line”, “Horizontal Bar”, “Vertical Bar”, and “Pie” chart.
## Chats Dataset:
The Given train Dataset has 2 directories one for train and other one for test with 5 different classes images their mapping stored csv.
## Steps to implement chat classification Model:
### Step 1: Import the libraries
TensorFlow, pathlib, matplotlib, numpy, pandas and tensorflow.keras libraries are used.
### Step 2: Load the dataset
At first, we imported the “os” module to interact with the operating system. Then we import “listdir()“ function from os to get access to the path.read one by one image in the directory and append to images array and mapped with the index and identified the correct chat label and converted that label in to number format.
### Step 3: visualize one instance of each class present in the dataset.
Implemented the code to view one image in each class. to see the pattern and understand the pattern of each digit.
### Step 4: Pre-process the data
Split the data into train and test with the test_size=0.2. The image data cannot be fed directly into the model, so we need to perform some operations and process the data to make it ready for our neural network. Converted the categorical chart label into numeric label by using label encoder.
### Step 5: check how many images we have of each class¶
implemented code to find the total number of images in each class of training dataset and have drawn plot to find the images in the train dataset is equally distributed or not. we can clearly visualize in the plot that we are almost equal number of images in each class. means data is evenly distributed.
### Step 6: Create the model:
I used the Kera’s Sequential API, where I just have added one layer at a time, starting from the input.

CNN model generally consists of convolutional and pooling layers. It works better for data that are represented as grid structures; this is the reason why CNN works well for image classification problems. The dropout layer is used to deactivate some of the neurons and while training, it reduces offer fitting of the model. We will then compile the model with the Adadelta optimizer.

The model has two main aspects: the feature extraction front end comprised of convolutional and pooling layers, and the classifier backend that will make a prediction.

For the convolutional front-end, we can start with a single convolutional layer with a small filter size (5,5) and a modest number of filters (60) followed by a max pooling layer. The filter maps can then be flattened to provide features to the classifier.

Given problem is a multi-class classification task, we know that we will require an output layer with 50 nodes in order to predict the probability distribution of an image belonging to each of the 5 classes. This will also require the use of a SoftMax activation function. Between the feature extractor and the output layer, we can add a dense layer to interpret the features, in this case with 500 nodes. Added Dropout to Avoid Overfitting.

We will use a conservative configuration for the stochastic gradient descent optimizer with a learning rate of 0.001 The categorical cross-entropy loss function will be optimized, suitable for multi-class classification, and we will monitor the classification accuracy metric, which is appropriate given we have the same number of examples in each of the 5 classes.
### Step :7 Compile, fit and Evaluate Model
Once the model has been created, it is time to compile it and fit the model. During the process of fitting, the model will go through the dataset and understand the relations. It will learn throughout the process as many times as has been defined. In our example, we have defined different epochs. During the process, the CNN model will learn and make mistakes. For every mistake (i.e., wrong predictions) the model makes, there is a penalty and that is represented in the loss value for each epoch. In short, the model should generate as little loss and as high accuracy as possible at the end of the last epoch.

 

I have implemented 4 different models to increase the accuracy. From the above image we can clearly see that model 4 is giving higher accuracy compared to the other models.
### Step 8: Plot the change in accuracy and loss per epochs¶
plotted the curve to check the variation of accuracy and loss as the number of epochs increases in each model for this I have used, matplotlib to plot the curve.
For each model we can clear see the increase in accuracy and decrease in loss.

 
### Step 9: Confusion Matrix for each model with validation data
 

### Step 10: Conclusion
I have implemented the Convolutional Neural Network for the classification of Gurumukhi dataset. Given dataset is divided into training and test dataset. It was taken to make predictions of handwritten digits from 0 to 9. The dataset was cleaned, scaled, and shaped. Using TensorFlow, a CNN model was created and was eventually trained on the training dataset. Finally, predictions were made using the trained model. Loss and accuracy plot for each epoch is implemented and confusion matrix is implemented for each model.
## References
•	Train Neural Network by loading your images |TensorFlow, CNN, Keras tutorial: https://www.youtube.com/watch?v=uqomO_BZ44g
•	YouTube- Text Detection using Neural Networks | OPENCV Python https://www.youtube.com/watch?v=y1ZrOs9s2QA

•	Deep Learning Project – Handwritten Digit Recognition using Python -https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/
•	Simple CNN keras (Accuracy : 0.99)=>Top 1% 
https://www.kaggle.com/code/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1
