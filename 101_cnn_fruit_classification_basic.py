#########################################################################
# Convolutional Neural Network - Fruit Classification
#########################################################################


#########################################################################
# Import required packages
#########################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense          # Conv1D: audio, Conv2D: Image, Conv3D: Video
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

#########################################################################
# Set Up flow For Training & Validation data
#########################################################################

# data flow parameters

training_data_dir = 'data/training'                  # as we deal with images, we don't want to read a whole csv file as it needs a large memory.
                                                     # dir: directory
                                                     # We have already put 60%, 30% and 10% of images in the training, validation and test folders, respectively.
validation_data_dir = 'data/validation'
batch_size = 32                                      # number of images in each batch
img_width = 128
img_height = 128
num_channels = 3                                     # as we have colorful images here
num_classes = 6                                      # as we want to classify 6 different types of fruits

# image generators and rescaling

training_generator = ImageDataGenerator(rescale = 1./255)                # rescale the pixels to expedite the processes such as gradient descent
                                                                         # 1./255: pixel values are between 0 and 255. 
                                                                         # 1./: returns float values as integer values would be the same for all values between 0 and 1
validation_generator = ImageDataGenerator(rescale = 1./255)

# image flows (send batches from our hard drive to the network)

training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),      # resize the images 
                                                      batch_size = batch_size,                    # how many images in each batch
                                                      class_mode = 'categorical')                 # type of problem that we want to solve (binary, categorical, ...). Here, we have multi-label classification

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')
#########################################################################
# Network Architecture
#########################################################################

# network architecture

model = Sequential()

    # first convolutional layer
model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
                                                                               # filters: number of feature maps (filters): 32 is just a random number to start with
                                                                               # kernel_size: filter size
                                                                               # stride: default is (1,1): number of pixels that we move across and then down in each iteration
                                                                               # padding: (valid (default): no padding will be added), same: padding will be added in a way that we can utilize all the pixels)
                                                                               # input_shape: dimension of input image, just required for the 1st conv layer
model.add(Activation('relu'))       
model.add(MaxPooling2D())                                                      # pool size: default is (2,2)

    # second convolutional layer 
model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'))        # this is the consequent layer that gets the info from the previous layer not the input layer
model.add(Activation('relu'))       
model.add(MaxPooling2D()) 

model.add(Flatten())                                                           # Flatten always comes after last conv layer

    # dense (fully connected) layer  
model.add(Dense(32))                                                           # 32: number of neurons that we want in our dense layer
model.add(Activation('relu'))       

    # output layer                                                                             
model.add(Dense(num_classes))                                                           
model.add(Activation('softmax'))                                               # gets the probabilities for all classes and add them to get a value of 1                                

# compile network

model.compile(loss = 'categorical_crossentropy',  optimizer = 'adam', metrics = ['accuracy'])

# view network architecture

model.summary()


#########################################################################
# Train Our Network!
#########################################################################

# training parameters

num_epochs = 50                                                                # 50 is a just random number and we may need to adjust it
                                                                               # We will pass our training images 50 times through the network
model_filename = 'models/fruits_cnn_v01.h5'                                    # to save our model named fruits_cnn_v01 in models forlder

# callbacks

save_best_model = ModelCheckpoint(filepath = model_filename, 
                                  monitor = 'val_accuracy',
                                  mode = 'max',                                # if it was 'val_loss', we have to go with 'min'
                                  verbose = 1,                                 # by default is zero but we set it to 1 to be notified if and when there is an update in our model (upoan having a model improvement, our model will be updated)
                                  save_best_only = True)  
                              
# train the network

history = model.fit(x = training_set, 
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])

#########################################################################
# Visualise Training & Validation Performance
#########################################################################

import matplotlib.pyplot as plt

# plot validation results to check for overfitting
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()                                                                     # An increase in the gap between (training_accuracy and val_accuracy) and (training_loss and val_loss) is a sign of overfitting: model learnt perfectly from the training data but does not perform well for validation set

# get best epoch performance for validation accuracy
max(history.history['val_accuracy'])


#########################################################################
# Make Predictions On New Data (Test Set)
#########################################################################

# import required packages

from tensorflow.keras.models import load_model                                 # to import the model file that we saved during training
from tensorflow.keras.preprocessing.image import load_img, img_to_array        # to import image file and convert it to sth that keras can work with
import numpy as np
import pandas as pd
from os import listdir                                                         # os: a module that allows us to work on files on our hard drive
                                                                               # listdir: list of files within any specified directory
                                                                  
# parameters for prediction

model_filename = 'models/fruits_cnn_v01.h5'                                    # to save our model named fruits_cnn_v01 in models forlder
img_width = 128                                 
img_height = 128                                                               # all images should be the same size otherwise, we get some errors
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']        # It should be in the same order that we have on our hard drive

# load model

model = load_model(model_filename)

# import an image & apply pre-processing

filepath = 'data/test/banana/banana_0074.jpg'          

image = load_img(filepath, target_size = (img_width, img_height))

    # convert image to array
image = img_to_array(image)                                                    # image.shape: (128,128,3): (image_width, image_height,RGB)

    # add size of image batch 
image= np.expand_dims(image, axis = 0)                                         # image.shape: batch of n images: (n, image_width, image_height,RGB)
                                                                               # axis = 0: to add the new dimension at the start of the array (here, n=1 is added at the begining)
    # normalize the pixel values 
image = image * (1./255)                                                       # convert to a range between 0 and 1 (1./ gives us an integer)

    # predict the probability of being in each class
class_probs = model.predict(image)

    # find the index of the class with the maximum probability
predicted_class = np.argmax(class_probs)

    # find the label of the predicted class
predicted_label = labels_list[predicted_class]

    # find the probability of the predicted class
predicted_prob = class_probs[0][predicted_class]

# image pre-processing function (alternative for previous approach)

def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)                                                    
    image= np.expand_dims(image, axis = 0)                                         
    image = image * (1./255) 
                                                       
    return image
  
# image prediction function (alternative for previous approach)

def make_prediction(image):
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    
    return predicted_label, predicted_prob

# test our recently defined function

image = preprocess_image(filepath)
make_prediction(image)

# loop through test data

source_dir = 'data/test/'
folder_names = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']
actual_labels = []
predicted_labels = []
predicted_probabilities = []
filenames = []

for folder in folder_names:
    
    images = listdir(source_dir + '/' + folder)
    
    for image in images:
        
        processed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_probability = make_prediction(processed_image)
        actual_labels.append(folder)
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(predicted_probability)
        filenames.append(image)
        
# create dataframe to analyse

predictions_df = pd.DataFrame({'actual_label': actual_labels,
                               'predicted_label': predicted_labels,
                               'predicted_probability': predicted_probabilities,
                               'filename': filenames})

predictions_df['correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_label'], 1, 0)      # If correct put 1, if not put 0


# overall test set accuracy

test_set_accuracy = predictions_df['correct'].sum()/len(predictions_df)        # devide number of 1s(correct predictions) by total number of predictions
print(test_set_accuracy)
# 80% (basic)

"""
If we open the prediction_df, we can find the item with the lowest predicted probability (the most confusing item for the network)
"""

# confusion matrix (raw numbers)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'])
print(confusion_matrix)

# confusion matrix (percentages)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)










