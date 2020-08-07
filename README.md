# Project Overview

The project is to create a Convolutional Neural Networks (CNN) from scratch and leverage the image classification techniques. Given an image of a bird, the model will identify an estimate of the breed. If supplied an image of a non-bird, the code will identify the resembling bird breed. This model can be used as part of a mobile or web app for the real world and user-provided images. That means the problem is about classification of data with labeled in supervised learning. Supervised learning is the task of learning a function that maps an input to an output. It infers a function from labelled training data consisting of a set of training dataset.

![image](https://github.com/michaelcity/Bird-Species-Recognition/blob/master/img/example1.JPG)

# Project Design

1.	Import Bird Breeds Dataset which provided by Kaggle (https://www.kaggle.com/gpiosenka/100-bird-species/kernels)

2.	Detects bird in the image by using VGG16 pre-trained model 

3.	Create a CNN to Classify Bird Breeds

4.	Create a transfer learning CNN to Classify Bird Breeds



# Project Instructions

1. Download Bird dataset from [Kaggle](https://www.kaggle.com/gpiosenka/100-bird-species) (224x224x3 in jpg format), it provided 30168 training images including 1125 test images(5 per species) and 1125 validation images(5 per species. Images for each species are contained in a separate sub director and all the files were numbered sequential. For example, AMERICAN GOLDFINCH/001.jpg , AMERICAN GOLDFINCH/003.jpg etc.
 
2. Make sure you have already installed the necessary Python packages according to the README in the program repository. Also you can run this note in colab or AWS SageMaker Notebook instance. [Anaconda](https://www.anaconda.com/) also recommamd for ML Tools.
If you running on local machine, you can start the jupyter notebook.

`jupyter notebook dog_app.ipynb`



# (Optionally) Accelerating the Training Process

To reduce the training time for complexity of CNN architecture, you can switch the code on a GPU. If you'd like to use a GPU, you can spin up an instance of your own:

Amazon SageMaker Notebook instance
You can use Amazon Web Services to launch an EC2 GPU instance. (This costs money)

Google Colab
Free for use and free GPU resource but not guarantee the performace
