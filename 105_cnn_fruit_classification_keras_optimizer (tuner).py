#########################################################################
# Convolutional Neural Network - Fruit Classification
#########################################################################


#########################################################################
# Import required packages
#########################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner.tuners import RandomSearch                                    # required for tuning
from keras_tuner.engine.hyperparameters import HyperParameters                 # required for tuning
import os                                                                      # to be able to work with files in the hard drive (just for windows users)

#########################################################################
# Set Up flow For Training & Validation data
#########################################################################

# data flow parameters

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32                                      # number of images in each batch
img_width = 128
img_height = 128
num_channels = 3                                     # as we have colorful images here
num_classes = 6                                      # as we want to classify 6 different types of fruits

# image generators

training_generator = ImageDataGenerator(rescale = 1./255,                # rescale the pixels to expedite the processes such as gradient descent
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5, 1.5),
                                        fill_mode = 'nearest')
                                                                         # 1./255: pixel values are between 0 and 255. 
                                                                         # 1./: returns float values as integer values would be the same for all values between 0 and 1
"""
Data augmentation is only applied on the training set not the validation or test set
"""

validation_generator = ImageDataGenerator(rescale = 1./255)

# image flows (send batches from our hard drive to the network)

training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')         # type of problem that we want to solve (binary, categorical, ...). Here, we have multi-label classification

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')


#########################################################################
# Architecture Tuning (trying different number of neurons)
#########################################################################

# network architecture

def build_model(hp):
    model = Sequential()

        # first convolutional layer
    model.add(Conv2D(filters = hp.Int('Input_Conv_Filters', min_value = 32, max_value = 128, step = 32), kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
                                                                                   # filters: number of feature maps (filters)
                                                                                   # hp.Int('Input_Conv_Filters'): we want our hp parameter to be integer and its name is Input_Conv_Filters
                                                                                   # min_value = 32, max_value = 128, step = 32: different filters to be tested
                                                                                   # kernel_size: filter size
                                                                                   # stride: default is (1,1)
                                                                                   # padding: (valid (default): no padding will be added), same: padding will be added in a way that we can utilize all the pixels)
                                                                                   # input_shape: dimension of input image
                                                                                   
    model.add(Activation('relu'))       
    model.add(MaxPooling2D()) 
    
    for i in range(hp.Int('n_Conv_Layers', min_value = 1, max_value = 3, step = 1)):     # test different number of conv layers

            # second convolutional layer 
        model.add(Conv2D(filters = hp.Int(f'Conv_{i}_Filters', min_value = 32, max_value = 128, step = 32), kernel_size = (3, 3), padding = 'same'))        # to change number of filters
        model.add(Activation('relu'))       
        model.add(MaxPooling2D()) 

    model.add(Flatten())  
    
    for j in range(hp.Int('n_Dense_Layers', min_value = 1, max_value = 4, step = 1)):     # test different number of dense layers


            # dense layer with dropout  
        model.add(Dense(hp.Int(f'Dense_{j}_Neurons', min_value = 32, max_value = 128, step = 32)))                                                           # to change number of neurons in each layer to test
        model.add(Activation('relu')) 
        
        if hp.Boolean('Dropout'):                                                          # If True, Dropout will be run
            model.add(Dropout(0.5))                                                        # 0.5: 50% of neurons will be dropped out (usually works well)

        # output layer                                                                             
    model.add(Dense(num_classes))                                                           
    model.add(Activation('softmax'))                                               # gets the probabilities for all classes and add them to get a value of 1                                

    # compile network

    model.compile(loss = 'categorical_crossentropy',  
                  optimizer = hp.Choice('Optimizer', values = ['adam', 'RMSProp']),     # instead of just adam optimizer, we can use hp and define a list of different optimizers
                  metrics = ['accuracy'])
    
    return model

# searcher

tuner = RandomSearch(hypermodel = build_model,                                          # the model function that the tuner will work with
                     objective = 'val_accuracy',                                        # what we want the tuner optimize
                     max_trials = 3,                                                    # maximum trials to achieve the optimized parameter
                     executions_per_trial = 2,                                          # number of times that each trial will be executed and the val_accuracy that is used to assess the performance would be the average of all of runs (n trials*m execution per trial)
                     directory = os.path.normpath('C:/'),                               # select the path to save the result. We will save the file to the base of our file system 
                     project_name = 'fruit-cnn',                                        # name of folder that we will create on the above path
                     overwrite = True)                                                  # updated results will be saved on the old ones

tuner.search(x = training_set,
             validation_data = validation_set,
             epochs = 5,                                                                # number of epochs for each test
             batch_size = 32)           

# top network (the best configuration that we got)
tuner.results_summary()   

# best network - hyperparameters
tuner.get_best_hyperparameters()[0].values   

# summary of best network architecture
tuner.get_best_models()[0].summary()


#########################################################################
# Network Architecture
#########################################################################

# network architecture

model = Sequential()

"""
In previous section, we found out that the best model has 3 conv (with 96, 64 and 64 neurons), 3 pooling and 1 dense (with 160 neurons) layers
"""

    # first convolutional layer (number of filters adjusted based on the tuning section)
model.add(Conv2D(filters = 96, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
                                                                               # filters: number of feature maps (filters): 32 is just a random number to start with
                                                                               # kernel_size: filter size
                                                                               # stride: default is (1,1)
                                                                               # padding: (valid (default): no padding will be added), same: padding will be added in a way that we can utilize all the pixels)
                                                                               # input_shape: dimension of input image
model.add(Activation('relu'))       
model.add(MaxPooling2D()) 

    # second convolutional layer (number of filters adjusted based on the tuning section)
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))        # this is the consequent layer that gets the info from the previous layer not the input layer
model.add(Activation('relu'))       
model.add(MaxPooling2D()) 

    # third convolutional layer (number of filters adjusted based on the tuning section)
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))        # this is the consequent layer that gets the info from the previous layer not the input layer
model.add(Activation('relu'))       
model.add(MaxPooling2D()) 

model.add(Flatten())  

    # dense layer with dropout (number of filters adjusted based on the tuning section)  
model.add(Dense(160))                                                           # 32: number of neurons that we want in our dense layer
model.add(Activation('relu')) 
model.add(Dropout(0.5))                                                        # 0.5: 50% of neurons will be dropped out (usually works well)

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

num_epochs = 50                                                                # 50 is just a random number and we may need to adjust it
model_filename = 'models/fruits_cnn_tuned.h5'                                    # to save our model named fruits_cnn_v01 in models forlder

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

# plot validation results
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# get best epoch performance for validation accuracy
max(history.history['val_accuracy'])


#########################################################################
# Make Predictions On New Data (Test Set)
#########################################################################

# import required packages

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from os import listdir                                                         # os: a module that allows us to work on files on our hard drive
                                                                               # listdir: list of files within any specified directory
                                                                  
# parameters for prediction

model_filename = 'models/fruits_cnn_tuned.h5'                                    # to save our model named fruits_cnn_v01 in models forlder
img_width = 128                                 
img_height = 128                                                               # all images should be the same size otherwise, we get some errors
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']

# load model

model = load_model(model_filename)

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
# 85% (dropout)
# 98% (augmentation)        model.add(Dropout(0.5)) was removed in the dense layer
# 95% (dropout+augmentation)

"""
If we open the prediction_df, we can find the item with the lowest predicted probability (the most confusing item for the network)
"""

# confusion matrix (raw numbers)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'])
print(confusion_matrix)

# confusion matrix (percentages)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)










 