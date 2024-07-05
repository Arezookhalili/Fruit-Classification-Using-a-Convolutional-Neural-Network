#########################################################################
# Convolutional Neural Network - Fruit Classification
#########################################################################


#########################################################################
# Import required packages
#########################################################################

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

#########################################################################
# Set Up flow For Training & Validation data
#########################################################################

# data flow parameters

training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32                                      # number of images in each batch
img_width = 224                                      # required for VGG model
img_height = 224
num_channels = 3                                     # as we have colorful images here
num_classes = 6                                      # as we want to classify 6 different types of fruits

# image generators

training_generator = ImageDataGenerator(preprocessing_function = preprocess_input,                # instead of rescale, we use preprocess_input that we already imported from VGG module
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

validation_generator = ImageDataGenerator(preprocessing_function = preprocess_input)               # instead of rescale, we use preprocess_input that we already imported from VGG module

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
# Network Architecture
#########################################################################

# network architecture

vgg = VGG16(input_shape = (img_width, img_height, num_channels), include_top = False)         # include_top = False: We just need the conv and the pooling layers and not the output and dense layer from vgg
                                                                                              # vgg has a massive dense layer and using transfer learning we can skip over them and save time and space

# vgg.summary()                                                                               # shows number of all parameters including trainable ones and we will see how they change when we apply transfer learning
                                                                                              
# freeze all layers (they won't be updated during training as we will just update the dense layer)    

for layer in vgg.layers:
    layer.trainable = False                                                                   # number of trainable parameters will be set to zero                                         

flatten = Flatten()(vgg.output)                                                               # we will flatten vgg output to be imported to the newly developed dense layer

    # add first dense layer
dense1 = Dense(128, activation = 'relu')(flatten)                                             # 128 neurons is considered for the dense layer
                                                                                              # activation is set to relu as the same cativation function has been used in the original vgg
                                                                                              # the input of this layer comes from the flatten layer

    # add second dense layer
dense2 = Dense(128, activation = 'relu')(dense1)

    # add output layer
output = Dense(num_classes, activation = 'softmax')(dense2)                                   # number of neurons in the output layer equals to number of classes that we already defined


model = Model(inputs = vgg.input, outputs = output)                                           # our model gets its input from original vgg model

             
# compile network

model.compile(loss = 'categorical_crossentropy',  optimizer = 'adam', metrics = ['accuracy'])

# view network architecture

model.summary()


#########################################################################
# Train Our Network!
#########################################################################

# training parameters

num_epochs = 10                                                                # 10 is just a random number and we may need to adjust it (as vgg is a very good model, 10 epochs should be good)
model_filename = 'models/fruits_cnn_vgg.h5'                                    # to save our model named fruits_cnn_v01 in models forlder

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

model_filename = 'models/fruits_cnn_vgg.h5'                                    # to save our model named fruits_cnn_v01 in models forlder
img_width = 224                                 
img_height = 224                                                               # as we use vgg, it should be 224
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']

# load model

model = load_model(model_filename)

# image pre-processing function (alternative for previous approach)

def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)                                                    
    image= np.expand_dims(image, axis = 0)                                         
    image = preprocess_input(image)                                            # instead of rescaling, we use what we already imported for vgg
                                                       
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
# 95% (transfer learning)


"""
If we open the prediction_df, we can find the item with the lowest predicted probability (the most confusing item for the network)
"""

# confusion matrix (raw numbers)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'])
print(confusion_matrix)

# confusion matrix (percentages)

confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)










 