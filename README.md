# Bird-Species-Recognition

# Domain Background
___
When I wake up at every morning, I can see the birds standing in front of my windows. Unfortunately, I cannot recognize the species which they belong to. That is my ideal of my project Bird species recognition.  That means the problem is about classification of data with labeled in supervised learning. Supervised learning is the task of learning a function that maps an input to an output. It infers a function from labelled training data consisting of a set of training dataset.


# Problem Statement
The project is to create a Convolutional Neural Networks (CNN) from scratch and leverage the image classification techniques This model can be used as part of a mobile or web app for the real world and user-provided images. Given an image to the model, it determines if a bird is present and returns the estimated breed. To achieve the goal,  the task involved the following step:
1)	download the data set and 
2)	detect the bird object from per trained model.
3)	 To achieve the high accuracy, using the pre trained model to train the dataset is a must.
4)	Classify the image by input simple image.


# Datasets and Input
___
By using provide Bird dataset from Kaggle(224x224x3 in jpg format), it provided 30168 training images including 1125 test images(5 per species) and 1125 validation images(5 per species. Images for each species are contained in a separate sub director and all the files were numbered sequential. For example, AMERICAN GOLDFINCH/001.jpg , AMERICAN GOLDFINCH/003.jpg etc.
 

 Also, the dataset contains imbalance gender ratio, 80% of the images are of the male and 20% of the female. Such that, this project will not recognize the gender of the species.
Furthermore, I decided filter out some of data set dataset, because the size was huge and some of bird species is not appear in my place (Hong Kong). 


# Solution Statement
The bird image need to normalize by using data normalization(224px x 224px), that is an important pre-processing step of the images. It ensures that each input comes from a standard distribution. The outcome is makes the model train faster. By using Convolutional Network for Classification and Detection, it is suitable for feature learning automatically, is good to find the feather in the image on each level. Finally, the layer uses the generated feature for classification.
Benchmark Model
 
 ![image](https://github.com/michaelcity/Bird-Species-Recognition/blob/master/img/kaggle-caputre.PNG)
 
Figure 2 Pre-trained model information form https://keras.io/api/applications/

The Figure 1 is about deep learning models that are made available alongside pre-trained weights from keras. By using the pre-trained model, the accuracy of image classification is up to 70%, VGG16 will be used for this project because of the dataset image size was 224x224 which suitable for VGG16 input.


 ![image](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)
 
Figure 3 VGG16 image from https://neurohive.io/en/popular-networks/vgg16/

The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It makes the improvement over AlexNet by replacing large kernel-sized filters with multiple 3×3 kernel-sized filters one after another.


# Evaluation Metrics

Accuracy is used for evaluation metric and using the CrossEntropy loss function. First, the labels have to be in a categorical format. The target files are the list of encoded bird labels related to the image with this format. This multi-class log loss punishes the classifier if the predicted probability leads to a different label than the actual and cause higher accuracy.



# Project Design

1.	Import Bird Breeds Dataset which provided by Kaggle (https://www.kaggle.com/gpiosenka/100-bird-species/kernels)

2.	Detects bird in the image
•	Using VGG16 pre-trained model 

3.	Create a CNN to Classify Bird Breeds
•	Data Loader for birds dataset
•	Using VGG to train the dataset
•	Config CrossEntropyloss function
•	Config ReLu
•	Train the model and validate the model

4.	Create Algorithm to test the model on simple Images
