# covid-19-detector-deep-learning
Basically i took datasets from two sites.I took normal chest x ray images from github and chest x ray of covid suffered patients from kaggle.Then i createdf my final dataset of images containing separate folders for train and test using jupyter notebook.Now i moved to google colab in order to access its gpu version so that i can process my model fast.
Now using convolutional neural network, i created a model which takes chest x ray image as input and predict whether a patient is nowmal or is suffering from covid.
Then i deployed my model using flask framework in pycharm which creates http request for our final application.
