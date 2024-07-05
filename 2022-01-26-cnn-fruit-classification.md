---
layout: post
title: Fruit Classification Using A Convolutional Neural Network
image: "/posts/cnn-fruit-classification-title-img1.png"
tags: [Deep Learning, CNN, Data Science, Computer Vision, Python]
---

In this project, I would like to help a grocery retailer enhance and scale their sorting and delivery processes through building & optimizing a Convolutional Neural Network to classify images of fruits. 

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
- [01. Data Overview](#data-overview)
- [02. Data Pipeline](#data-pipeline)
- [03. CNN Overview](#cnn-overview)
- [04. Baseline Network](#cnn-baseline)
- [05. Tackling Overfitting With Dropout](#cnn-dropout)
- [06. Image Augmentation](#cnn-augmentation)
- [07. Hyper-Parameter Tuning](#cnn-tuning)
- [08. Transfer Learning](#cnn-transfer-learning)
- [09. Overall Results Discussion](#cnn-results)
- [10. Next Steps & Growth](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

A robotics company informed a grocery store that they had built a prototype for a robotic sorting arm that could be used to pick up and move products off a platform. It would use a camera to "see" the product and could be programmed to move that particular product into a designated bin, for further processing.

The only thing they hadn't figured out was how to identify each product using the camera so that the robotic arm could move it to the right place.

I was asked to put forward a proof of concept on this - and was given some sample images of fruits from their processing platform.

If this was successful and put into place on a larger scale, the client would be able to enhance their sorting & delivery processes.

<br>
### Actions <a name="overview-actions"></a>

I utilized the *Keras* Deep Learning library for this task.

I started by creating a pipeline for feeding training & validation images in batches, from my local directory, into the network. I investigated & quantified predictive performance epoch by epoch on the validation set, and then also on a held-back test set.

My baseline network was simple but gave me a starting point to refine. This network contained **2 Convolutional Layers**, each with **32 filters** and subsequent **Max Pooling** Layers.  I had a **single Dense (Fully Connected) layer** following flattening with **32 neurons** followed by an output layer. I applied the **relu** activation function on all layers and used the **adam** optimizer.

My first refinement was to add **Dropout** to tackle the issue of overfitting which was prevalent in the baseline network performance. I used a **dropout rate of 0.5**.

I then added **Image Augmentation** to my data pipeline to increase the variation of input images for the network to learn from, resulting in more robust results as well as addressing overfitting.

With these additions in place, I utilized *keras-tuner* to optimize my network architecture & tune the hyperparameters. The best network from this testing contained **3 Convolutional Layers**, each followed by **Max Pooling** Layers. The first Convolutional Layer had **96 filters**, and the second & third had **64 filters**. The output of this third layer was flattened and passed to a **single Dense (Fully Connected) layer** with **160 neurons**. The Dense Layer had **Dropout** applied with a **dropout rate of 0.5**. The output from this was passed to the output layer.  Again, I applied the **relu** activation function on all layers and used the **adam** optimizer.

Finally, I utilized **Transfer Learning** to compare my network's results against that of the pre-trained **VGG16** network.

___
<br>
# Data Overview  <a name="data-overview"></a>

To build out this proof of concept, the client provided me with some sample data. This was made up of images of six different types of fruit, sitting on the landing platform in the warehouse.

I randomly splitted the images for each fruit into training (60%), validation (30%) and test (10%) sets.

Examples of four images of each fruit class can be seen in the image below:

![alt text](/img/posts/cnn-image-examples.png "CNN Fruit Classification Samples")

<br>
For ease of use in Keras, my folder structure first splitted into training, validation, and test directories, and within each of those was splitted again into directories based upon the six fruit classes.

All images were of size 300 x 200 pixels.

___
<br>
# Data Pipeline  <a name="data-pipeline"></a>

Before I got to building the network architecture, and subsequently training & testing it - I needed to set up a pipeline for my images to flow through, from my local hard drive where they were located, to, and through my network.

In the code below, I:

* Imported the required packages
* Set up the parameters for my pipeline
* Set up my image generators to process the images as they came in
* Set up my generator flow - specifying what I wanted to pass in for each iteration of training

```python
# Import the required Python libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
```
<br>
I specified that I would resize the images down to 128 x 128 pixels and that I would pass in 32 images at a time (known as the batch size) for training.
<br>
```python
# Data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6
```
<br>
To start with, I simply used the generators to rescale the raw pixel values (ranging between 0 and 255) to float values that existed between 0 and 1. The reason I did this was mainly to help Gradient Descent found an optimal, or near optimal solution each time much more efficiently - in other words, it meant that the features that were learned in the depths of the network were of a similar magnitude, and the learning rate that was applied to descend the loss or cost function across many dimensions, was somewhat proportionally similar across all dimensions - and long story short, meant training time was faster as Gradient Descent could converge faster each time!
<br>
```python
# Image generators
training_generator = ImageDataGenerator(rescale = 1./255)
validation_generator = ImageDataGenerator(rescale = 1./255)
```
<br>
I then sent the image batches from my hard drive to the network.
<br>
```python
# Image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                                      target_size = (img_width, img_height),
                                                                      batch_size = batch_size,
                                                                      class_mode = 'categorical')
```
<br>
I added more logic to the training set generator to apply Image Augmentation.

With this pipeline in place, my images would be extracted, in batches of 32, from my hard drive, where they're being stored and sent into my model for training!

___
<br>
# Convolutional Neural Network Overview <a name="cnn-overview"></a>

Convolutional Neural Networks (CNN) are an adaptation of Artificial Neural Networks and are primarily used for image-based tasks.

To a computer, an image is simply made up of numbers, those being the color intensity values for each pixel.  Color images have values ranging between 0 and 255 for each pixel, but have three of these values, for each - one for Red, one for Green, and one for Blue, or in other words the RGB values that mix to make up the specific color of each pixel.

These pixel values are the *input* for a Convolutional Neural Network. It needs to make sense of these values to make predictions about the image, for example, in my task here, to predict what the image was of, one of the six possible fruit classes.

The pixel values themselves don't hold much useful information on their own - so the network needs to turn them into *features* much like we do as humans.

A big part of this process is called **Convolution** where each input image is scanned over, and compared to many different, and smaller filters, to compress the image down into something more generalized.  This process not only helps reduce the problem space, it also helps reduce the network's sensitivity to minor changes, in other words, to know that two images are of the same object, even though the images are not *exactly* the same.

A somewhat similar process called **Pooling** is also applied to facilitate this *generalization* even further.  A CNN can contain many of these Convolution & Pooling layers - with deeper layers finding more abstract features.

Activation Functions are also applied to the data as it moves forward through the network, helping the network decide which neurons will fire, or in other words, helping the network understand which neurons are more or less important for different features, and ultimately which neurons are more or less important for the different output classes.

Over time - as a Convolutional Neural Network trains, it iteratively calculates how well it is predicting on the known classes we pass it (known as the **loss** or **cost**, then heads back through in a process known as **Back Propagation** to update the parameters within the network, in a way that reduces the error, or in other words, improves the match between predicted outputs and actual outputs.  Over time, it learns to find a good mapping between the input data, and the output classes.

Many parameters can be changed within the architecture of a Convolutional Neural Network, as well as clever logic that can be included, all of which can affect the predictive accuracy.  

___
<br>
# Baseline Network <a name="cnn-baseline"></a>

### Network Architecture

My baseline network was simple but gave me a starting point to refine. This network contained **2 Convolutional Layers**, each with **32 filters** and subsequent **Max Pooling** Layers.  I had a **single Dense (Fully Connected) layer** following flattening with **32 neurons** followed by my output layer. I applied the **relu** activation function on all layers and used the **adam** optimizer.

```python
# Network architecture
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# View network architecture
model.summary()
```
<br>
The below shows us my baseline architecture more clearly:
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 128, 128, 32)      896       
_________________________________________________________________
activation (Activation)      (None, 128, 128, 32)      0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 64, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 32)        9248      
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 32768)             0         
_________________________________________________________________
dense (Dense)                (None, 32)                1048608   
_________________________________________________________________
activation_2 (Activation)    (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 198       
_________________________________________________________________
activation_3 (Activation)    (None, 6)                 0         
=================================================================
Total params: 1,058,950
Trainable params: 1,058,950
Non-trainable params: 0
```

<br>
### Training The Network

With the pipeline, and architecture in place - I trained the baseline network!

In the below code, I:

* Specified the number of epochs for training
* Set a location for the trained network to be saved (architecture & parameters)
* Set a *ModelCheckPoint* callback to save the best network at any point during training (based upon validation accuracy)
* Trained the network and saved the results to an object called *history*

```python
# Training parameters
num_epochs = 50
model_filename = 'models/fruits_cnn_v01.h5'

# Callbacks
save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1,
                                  save_best_only = True)

# Train the network
history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])
```
<br>
The ModelCheckpoint callback that was put in place meant that I did not just save the *final* network at epoch 50, but instead, I saved the *best* network, in terms of validation set performance - from *any point* during training. In other words, at the end of each of the 50 epochs, Keras would assess the performance on the validation set and if it did not see any performance improvement it would do nothing. If it saw an improvement, however, it would update the network file that was saved on our hard drive.

<br>
### Analysis Of Training Results

As I saved my training process to the *history* object, I could analyze the performance (Classification Accuracy, and Loss) of the network epoch by epoch.

```python
Import matplotlib.pyplot as plt

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

# Get best epoch performance for validation accuracy
max(history.history['val_accuracy'])
```
<br>
The below image contains two plots, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second showing the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

![alt text](/img/posts/cnn-baseline-accuracy-plot.png "CNN Baseline Accuracy Plot")

<br>
There were two key learnings from the above plots. 
* With this baseline architecture & the parameters I set for training, I was reaching the best performance in around 10-20 epochs - after that, not much improvement was seen.  This wasn't to say that 50 epochs were wrong, especially if I change my network - but was interesting to note at this point.

* The significant gap between the orange and blue lines on the plot, in other words between my validation performance and my training performance. This gap was over-fitting.

Focusing on the lower plot above (Classification Accuracy) - it appeared that my network was learning the features of the training data *so well* that after about 20 or so epochs it was *perfectly* predicting those images - but on the validation set, it never passed approximately **83% Classification Accuracy**.

I did not want over-fitting! It meant that I was risking my predictive performance on new data. The network was not learning to generalize, meaning that if something slightly different came along then it's going to struggle to predict well, or at least predict reliably!

I tried to address this with some clever concepts in the next sections.

<br>
### Performance On The Test Set

Above, I assessed my model's performance on both the training set and the validation set - both of which were being passed in during training.

Here, I got a view of how well my network performed when predicting data that was *no part* of the training process whatsoever (my test set).

A test set can be extremely useful when looking to assess many different iterations of a network we build.  

Where the validation set might be sent through the model in slightly different orders during training to assess the epoch-by-epoch performance, my test set was a *static set* of images. Because of this, it made for a really good baseline for testing the first iteration of my network versus any subsequent versions that I created, perhaps after I refined the architecture, or added any other clever bits of logic that I thought might help the network perform better in the real world.

In the below code, I ran this in isolation from training. I:

* Imported the required packages for importing & manipulating my test set images
* Set up the parameters for the predictions
* Loaded in the saved network file from training
* Created a function for preprocessing my test set images in the same way that training & validation images were
* Created a function for making predictions, returning both the predicted class label, and predicted class probability
* Iterated through my test set images, preprocessing each and passing to the network for prediction
* Created a Pandas DataFrame to hold all prediction data

```python
# Import required packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from os import listdir

# Parameters for prediction
model_filename = 'models/fruits_cnn_v01.h5'
img_width = 128
img_height = 128
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']

# Load model
model = load_model(model_filename)

# Image pre-processing function
def preprocess_image(filepath):
    
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image * (1./255)
    
    return image

# Image prediction function
def make_prediction(image):
    
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    
    return predicted_label, predicted_prob

# Loop through test data
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
        
# Create dataframe to analyze
predictions_df = pd.DataFrame({"actual_label" : actual_labels,
                               "predicted_label" : predicted_labels,
                               "predicted_probability" : predicted_probabilities,
                               "filename" : filenames})

predictions_df['correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_label'], 1, 0)
```
<br>
After running the code above, I ended up with a Pandas DataFrame containing prediction data for each test set image. A random sample of this can be seen in the table below:


| **actual_label** | **predicted_label** | **predicted_probability** | **filename** | **correct** |
|---|---|---|---|---|
| apple | lemon | 0.700764 | apple_0034.jpg | 0 |
| avocado | avocado | 0.99292046 | avocado_0074.jpg | 1 |
| orange | orange | 0.94840413 | orange_0004.jpg | 1 |
| banana | lemon | 0.87131584 | banana_0024.jpg | 0 |
| kiwi | kiwi | 0.66800004 | kiwi_0094.jpg | 1 |
| lemon | lemon | 0.8490372 | lemon_0084.jpg | 1 |

<br>
In my data, I had:

* Actual Label: The true label for that image
* Prediction Label: The predicted label for the image (from the network)
* Predicted Probability: The network's perceived probability for the predicted label
* Filename: The test set image on our local drive (for reference)
* Correct: A flag showing whether the predicted label is the same as the actual label

This dataset was extremely useful as I could not only calculate my classification accuracy, but I could also deep-dive into images where the network was struggling to predict and try to assess why - leading to improving my network, and potentially my input data!

<br>
### Test Set Classification Accuracy

Using my DataFrame, I could calculate my overall Test Set classification accuracy using the below code:

```python
# Overall test set accuracy
test_set_accuracy = predictions_df['correct'].sum() / len(predictions_df)
print(test_set_accuracy)
```
<br>
My baseline network achieved a **75% Classification Accuracy** on the Test Set.  It would be interesting to see how much improvement I could see with these additions and refinements to my network.

<br>
### Test Set Confusion Matrix

Overall Classification Accuracy was very useful, but it could hide what was going on with the network's predictions!

As we saw above, My Classification Accuracy for the whole test set was 75%, but it might be that my network was predicting extremely well on apples, but struggling with Lemons as for some reason it was regularly confusing them with Oranges. A Confusion Matrix could help me uncover insights like this!

I could create a Confusion Matrix with the below code:

```python
# Confusion matrix (percentages)
confusion_matrix = pd.crosstab(predictions_df['predicted_label'], predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)
```
<br>
This resulted in the following output:

```
actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              0.8      0.0     0.0   0.1    0.0     0.1
avocado            0.0      1.0     0.0   0.0    0.0     0.0
banana             0.0      0.0     0.2   0.1    0.0     0.0
kiwi               0.0      0.0     0.1   0.7    0.0     0.0
lemon              0.2      0.0     0.7   0.0    1.0     0.1
orange             0.0      0.0     0.0   0.1    0.0     0.8
```
<br>
Along the top were my *actual* classes and down the side were my *predicted* classes - so by counting *down* the columns I could get the Classification Accuracy (%) for each class, and I could see where it was getting confused.

So, while overall my test set accuracy was 75%, I saw the following accuracies for each class:

* Apple: 80%
* Avocado: 100%
* Banana: 20%
* Kiwi: 70%
* Lemon: 100%
* Orange: 80%

This was very powerful as I could see what exactly was driving my *overall* Classification Accuracy.

The standout insight here was for Bananas - with a 20% Classification Accuracy, and even more interestingly I could see where it is getting confused. The network predicted 70% of Banana images to be of the class Lemon!

___
<br>
# Tackling Overfitting With Dropout <a name="cnn-dropout"></a>

<br>
### Dropout Overview

Dropout is a technique used in Deep Learning primarily to reduce the effects of overfitting. Over-fitting is where the network learns the patterns of the training data so specifically, that it runs the risk of not generalizing well, and being very unreliable when used to predict new, unseen data.

Dropout works in a way where, for each batch of observations that is sent forward through the network, a pre-specified proportion of the neurons in a hidden layer are essentially ignored or deactivated.  This can be applied to any number of the hidden layers.

When we apply Dropout, the deactivated neurons are completely taken out of the picture - they take no part in the passing of information through the network.

All the math is the same, the network will process everything as it always would (so taking the sum of the inputs multiplied by the weights, adding a bias term, applying activation functions, and updating the network’s parameters using Back Propagation) - it’s just that in this scenario where we are disregarding some of the neurons, we’re essentially pretending that they’re not there.

In a full network (i.e. where Dropout is not being applied) each of the combinations of neurons becomes quite specific in what it represents, at least in terms of predicting the output.  At a high level, if we were classifying pictures of cats and dogs, there might be some linked combination of neurons that fires when it sees pointy ears and a long tongue.  This combination of neurons becomes very tuned into its role in prediction, and it becomes very good at what it does - but as is the definition of overfitting, it becomes too good - it becomes too rigidly aligned with the training data.

If we *drop out* neurons during training, *other* neurons need to jump in and fill in for this particular role of detecting those features.  They essentially have to come in at late notice and cover the ignored neurons' job, dealing with that particular representation that is so useful for prediction.

Over time, with different combinations of neurons being ignored for each mini-batch of data - the network becomes more adept at generalizing and thus is less likely to overfit to the training data.  Since no particular neuron can rely on the presence of other neurons, and the features that they represent - the network learns more robust features, and are less susceptible to noise.

In a Convolutional Neural Network, it is generally best practice to only apply Dropout to the Dense (Fully Connected) layer or layers, rather than to the Convolutional Layers.  


<br>
### Updated Network Architecture

Here, I only had *one Dense Layer*, so I applied *Dropout to that layer only*. A common proportion to apply (i.e. the proportion of neurons in the layer to be deactivated randomly each pass) is 0.5 or 50%.  I applied this here.

```python
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
```

<br>
### Training The Updated Network

I ran the same code to train this updated network as I did for the baseline network (50 epochs) - the only change was that I modified the filename for the saved network to ensure I had all network files for comparison.

<br>
#### Analysis Of Training Results

As I again saved my training process to the *history* object, I could analyze & plot the performance (Classification Accuracy, and Loss) of the updated network epoch by epoch.

With the baseline network, I saw very strong overfitting in action - it would be interesting to see if the addition of Dropout was helpful!

The below image shows the same two plots I analyzed for the updated network, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second showing the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

![alt text](/img/posts/cnn-dropout-accuracy-plot.png "CNN Dropout Accuracy Plot")

<br>
Firstly, I could see a peak Classification Accuracy on the validation set of around **89%** which was higher than the **83%** I saw for the baseline network.

Secondly, and what I was looking to see, was that the gap between the Classification Accuracy on the training set, and the validation set was mostly eliminated. The two lines were trending up at more or less the same rate across all epochs of training - and the accuracy on the training set also never reached 100% as it did before meaning that we were indeed seeing this *generalization* that we wanted!

The addition of Dropout appeared to have remedied the overfitting that I saw in the baseline network. This was because, while some neurons were turned off during each mini-batch iteration of training - all would have their turn, many times, to be updated - just in a way where no neuron or combination of neurons would become so hard-wired to certain features found in the training data!

<br>
### Performance On The Test Set

During training, I assessed our updated network performance on both the training set and the validation set. Here, as I did for the baseline network, I would get a view of how well my network performed when predicting data that was *no part* of the training process whatsoever (my test set).

I ran the same code as I did for the baseline network, with the only change being to ensure I was loading in the network file for the updated network

<br>
### Test Set Classification Accuracy

My baseline network achieved a **75% Classification Accuracy** on the test set.  With the addition of Dropout, I saw both a reduction in overfitting and an increased *validation set* accuracy.  On the test set, I again saw an increase vs. the baseline, with an **85% Classification Accuracy**. 

<br>
### Test Set Confusion Matrix

As mentioned above, while overall Classification Accuracy was very useful, it could hide what was going on with the network's predictions!

The standout insight for the baseline network was that Bananas have only a 20% Classification Accuracy, very frequently being confused with Lemons. It would be interesting to see if the extra *generalization* forced upon the network with the application of Dropout helps this.

Running the same code from the baseline section on resulted for my updated network, I got the following output:

```
actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              0.8      0.0     0.0   0.0    0.0     0.0
avocado            0.0      1.0     0.1   0.2    0.0     0.0
banana             0.0      0.0     0.7   0.0    0.0     0.0
kiwi               0.2      0.0     0.0   0.7    0.0     0.1
lemon              0.0      0.0     0.2   0.0    1.0     0.0
orange             0.0      0.0     0.0   0.1    0.0     0.9
```
<br>
Along the top are my *actual* classes and down the side are my *predicted* classes - so by counting *down* the columns I can get the Classification Accuracy (%) for each class, and I can see where it is getting confused.

So, while overall my test set accuracy was 85% - for each class I saw:

* Apple: 80%
* Avocado: 100%
* Banana: 70%
* Kiwi: 70%
* Lemon: 100%
* Orange: 90%

All classes here were being predicted *at least* as well as with the baseline network - and Bananas which had only a 20% Classification Accuracy last time, were classified correctly 70% of the time.  Still the lowest of all classes, but a significant improvement over the baseline network!

___
<br>
# Image Augmentation <a name="cnn-augmentation"></a>

<br>
### Image Augmentation Overview

Image Augmentation is a concept in Deep Learning that aims to not only increase predictive performance but also to increase the robustness of the network through regularisation.

Instead of passing in each of the training set images as it stands, with Image Augmentation, we pass in many transformed *versions* of each image. This results in increased variation within our training data (without having to explicitly collect more images) meaning the network has a greater chance to understand and learn the objects we’re looking to classify, in a variety of scenarios.

Common transformation techniques are:

* Rotation
* Horizontal/Vertical Shift
* Shearing
* Zoom
* Horizontal/Vertical Flipping
* Brightness Alteration

When applying Image Augmentation using Keras' ImageDataGenerator class, we do this "on the fly" meaning the network does not train on the *original* training set image, but instead on the generated/transformed *versions* of the image - and this version changes each epoch. In other words - for each epoch that the network is trained, each image will be called upon, and then randomly transformed based upon the specified parameters - and because of this variation, the network learns to generalize a lot better for many different scenarios.

<br>
### Implementing Image Augmentation

I applied the Image Augmentation logic into the ImageDataGenerator class that existed within my Data Pipeline.

It is important to note that we only do this for our training data, we don't apply any transformation on our validation or test sets. The reason for this is that we want our validation & test data to be static and serve us better for measuring our performance over time. If the images in these sets kept changing because of transformations it would be really hard to understand if our network was improving, or if it was just a lucky set of validation set transformations that made it appear that it was performing better!

When setting up and training the baseline & Dropout networks - I used the ImageGenerator class for only one thing, to rescale the pixel values. Here, I added in the Image Augmentation parameters as well, meaning that as images flow into my network for training the transformations would be applied.

In the code below, I added these transformations and specified the magnitudes that I wanted to be applied:

```python
# Image generators
training_generator = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5,1.5),
                                        fill_mode = 'nearest')

validation_generator = ImageDataGenerator(rescale = 1./255)
```
<br>
**rotation_range** of 20: This was the *degrees* of rotation, and it dictated the *maximum* amount of rotation that I wanted. In other words, a rotation value would be randomly selected for each image, each epoch, between negative and positive 20 degrees.

**width_shift_range** and **height_shift_range** of 0.2:  I was allowing Keras to shift my image *up to* 20% both vertically and horizontally.

**zoom_range** of 0.1: meaning a maximum of 10% inward or outward zoom.

**horizontal_flip** to be True: meaning that each time an image flew in, there was a 50/50 chance of it being flipped.

**brightness_range** between 0.5 and 1.5: meaning that my images can become brighter or darker.

**fill_mode** set to "nearest": meaning that when images were shifted and/or rotated, we would just use the *nearest pixel* to fill in any new pixels that were required - and it meant that images still resembled the scene, generally speaking!

Again, it is important to note that these transformations were applied *only* to the training set, and not the validation set.

<br>
### Updated Network Architecture

My network would be the same as the baseline network.  I would not apply Dropout here to ensure I could understand the true impact of Image Augmentation for my task.

<br>
### Training The Updated Network

I ran the same code to train this updated network as I did for the baseline network (50 epochs) - the only change was that I modified the filename for the saved network to ensure I had all network files for comparison.

<br>
### Analysis Of Training Results

As I saved my training process to the *history* object, I could analyze & plot the performance (Classification Accuracy, and Loss) of the updated network epoch by epoch.

The below image shows the same two plots I analyzed for the updated network, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second showing the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

![alt text](/img/posts/cnn-augmentation-accuracy-plot.png "CNN Dropout Accuracy Plot")

<br>
Firstly, I could see a peak Classification Accuracy on the validation set of around **97%** which is higher than the **83%** I saw for the baseline network, and higher than the **89%** I saw for the network with Dropout added.

Secondly, and what I was really looking to see, was that the gap between the Classification Accuracy on the training set, and the validation set was mostly eliminated. The two lines were trending up at more or less the same rate across all epochs of training - and the accuracy on the training set also never reached 100% as it did before meaning that Image Augmentation was also giving the network this *generalization* that I wanted!

The reason for this was that the network was getting a slightly different version of each image each epoch during training, meaning that while it's learning features, it couldn't cling to a *single version* of those features!

<br>
### Performance On The Test Set

During training, I assessed my updated network performance on both the training set and the validation set.  Here, like I did for the baseline & Dropout networks, I would get a view of how well my network performs when predicting on data that was *no part* of the training process whatsoever (my test set).

I ran the same code as I did for the earlier networks, with the only change being to ensure I was loading in the network file for the updated network

<br>
### Test Set Classification Accuracy

My baseline network achieved a **75% Classification Accuracy** on the test set, and my network with Dropout applied achieved **85%**.  With the addition of Image Augmentation, I saw both a reduction in overfitting and an increased *validation set* accuracy. On the test set, i saw an increase vs. the baseline & Dropout, with a **93% Classification Accuracy**. 

<br>
### Test Set Confusion Matrix

As mentioned above, while overall Classification Accuracy was very useful, it could hide what was going on with the network's predictions!

The standout insight for the baseline network was that Bananas had only a 20% Classification Accuracy, very frequently being confused with Lemons. Dropout, through the additional *generalization* forced upon the network, helped a lot - let's see how my network with Image Augmentation fared!

Running the same code from the baseline section on results for my updated network, I got the following output:

```
actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              0.9      0.0     0.0   0.0    0.0     0.0
avocado            0.0      1.0     0.0   0.0    0.0     0.0
banana             0.1      0.0     0.8   0.0    0.0     0.0
kiwi               0.0      0.0     0.0   0.9    0.0     0.0
lemon              0.0      0.0     0.2   0.0    1.0     0.0
orange             0.0      0.0     0.0   0.1    0.0     1.0
```
<br>
Along the top are my *actual* classes and down the side are my *predicted* classes - so by counting *down* the columns I could get the Classification Accuracy (%) for each class, and I could see where it was getting confused.

So, while overall my test set accuracy was 93% - for each class I saw:

* Apple: 90%
* Avocado: 100%
* Banana: 80%
* Kiwi: 90%
* Lemon: 100%
* Orange: 100%

All classes here were predicted *more accurately* when compared to the baseline network, and *at least as accurate or better* when compared to the network with Dropout added.

Utilizing Image Augmentation *and* applying Dropout would be a powerful combination!

___
<br>
# Hyper-Parameter Tuning <a name="cnn-tuning"></a>

<br>
### Keras Tuner Overview

So far, with my Fruit Classification task, I had:

* Started with a baseline model
* Added Dropout to help with overfitting
* Utilised Image Augmentation

The addition of Dropout and Image Augmentation boosted both performance and robustness - but there was something that *could* have a big impact on how well the network learnt to find and utilize important features for classifying our fruits - and that was the network *architecture*!

So far, I had just used 2 convolutional layers, each with 32 filters, and I had used a single Dense layer, also, just by coincidence, with 32 neurons - and I admitted that this was just a place to start, my baseline.

One way for me to figure out if there were *better* architectures would be to just try different things. Maybe I just doubled my number of filters to 64, or maybe I kept the first convolutional layer at 32 but increased the second to 64. Perhaps I put a whole lot of neurons in my hidden layer, and then, what about things like use of Adam as an optimizer, was this the best one for my particular problem, or should I use something else?

I could start testing all of these things, and noting down performances, but that would be quite messy. Here, I would instead utilize *Keras Tuner* which would make this a whole lot easier for me!

At a high level, with Keras Tuner, I would ask it to test, a whole host of different architecture and parameter options, based upon some specifications that I put in place.  It would go off and run some tests, and return us with all sorts of interesting summary statistics, and of course information about what worked best.

Once I had this, I could then create that particular architecture, train the network just as I'd always done - and analyze the performance against my original networks.

My data pipeline would remain the same as it was when applying Image Augmentation. The code below showed this, as well as the extra packages I needed to load for Keras-Tuner.

```python
# Import the required Python libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import os

# Data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6

# Image generators
training_generator = ImageDataGenerator(rescale = 1./255)
validation_generator = ImageDataGenerator(rescale = 1./255)

# Image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                                      target_size = (img_width, img_height),
                                                                      batch_size = batch_size,
                                                                      class_mode = 'categorical')
```

<br>
### Application Of Keras Tuner

Here, I specified what I wanted Keras Tuner to test, and how I wanted it to test it!

I put my network architecture into a *function* with a single parameter called *hp* (hyperparameter)

I then made use of several in-build bits of logic to specify what I wanted to test.  In the code below, I tested for:

* Convolutional Layer Count - Between 1 & 4
* Convolutional Layer Filter Count - Between 32 & 256 (Step Size 32)
* Dense Layer Count - Between 1 & 4
* Dense Layer Neuron Count - Between 32 & 256 (Step Size 32)
* Application Of Dropout - Yes or No
* Optimizer - Adam or RMSProp

```python
# Network architecture
def build_model(hp):
    model = Sequential()
    
    model.add(Conv2D(filters = hp.Int("Input_Conv_Filters", min_value = 32, max_value = 256, step = 32), kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    for i in range(hp.Int("n_Conv_Layers", min_value = 1, max_value = 3, step = 1)):
    
        model.add(Conv2D(filters = hp.Int(f"Conv_{i}_Filters", min_value = 32, max_value = 256, step = 32), kernel_size = (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    for j in range(hp.Int("n_Dense_Layers", min_value = 1, max_value = 4, step = 1)):
    
        model.add(Dense(hp.Int(f"Dense_{j}_Neurons", min_value = 32, max_value = 256, step = 32)))
        model.add(Activation('relu'))
        
        if hp.Boolean("Dropout"):
            model.add(Dropout(0.5))
    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # Compile network
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = hp.Choice("Optimizer", values = ['adam', 'RMSProp']),
                  metrics = ['accuracy'])
    
    return model
```
<br>
Once I had the testing logic in place - I wanted to put in place the specifications for the search!

In the code below, I set the parameters to:

* Point to the network *function* with the testing logic (hypermodel)
* Set the metric to optimize for (objective)
* Set the number of random network configurations to test (max_trials)
* Set the number of times to try each tested configuration (executions_per_trial)
* Set the details for the output of logging & results

```python
# Search parameters
tuner = RandomSearch(hypermodel = build_model,
                     objective = 'val_accuracy',
                     max_trials = 30,
                     executions_per_trial = 2,
                     directory = os.path.normpath('C:/'),
                     project_name = 'fruit-cnn',
                     overwrite = True)
```
<br>
With the search parameters in place, I now wanted to put this into action.

In the below code, I:

* Specified the training & validation flows
* Specified the number of epochs for each tested configuration
* Specified the batch size for training

```python
# `Execute search
tuner.search(x = training_set,
             validation_data = validation_set,
             epochs = 40,
             batch_size = 32)
```
<br>
Depending on how many configurations were to be tested, how many epochs were required for each, and the speed of processing - this could take a long time, but the results would most definitely guide us toward a more optimal architecture!

<br>
### Updated Network Architecture

Based upon the tested network architectures, the best in terms of validation accuracy was one that contains **3 Convolutional Layers**. The first had **96 filters** and the subsequent two each **64 filters**.  Each of these layers had an accompanying MaxPooling Layer (this wasn't tested). The network then had **1 Dense (Fully Connected) Layer** following flattening with **160 neurons** with **Dropout applied** - followed by an output layer. The chosen optimizer was **Adam**.

```python
# Network architecture
model = Sequential()

model.add(Conv2D(filters = 96, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(160))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
```
<br>
The below shows us clearly my optimized architecture:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_10 (Conv2D)           (None, 128, 128, 96)      2688      
_________________________________________________________________
activation_20 (Activation)   (None, 128, 128, 96)      0         
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 64, 64, 96)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 64, 64, 64)        55360     
_________________________________________________________________
activation_21 (Activation)   (None, 64, 64, 64)        0         
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 32, 32, 64)        0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 32, 32, 64)        36928     
_________________________________________________________________
activation_22 (Activation)   (None, 32, 32, 64)        0         
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 16, 16, 64)        0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 16384)             0         
_________________________________________________________________
dense_10 (Dense)             (None, 160)               2621600   
_________________________________________________________________
activation_23 (Activation)   (None, 160)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 160)               0         
_________________________________________________________________
dense_11 (Dense)             (None, 6)                 966       
_________________________________________________________________
activation_24 (Activation)   (None, 6)                 0         
=================================================================
Total params: 2,717,542
Trainable params: 2,717,542
Non-trainable params: 0
```

<br>
My optimized architecture had a total of 2.7 million parameters, a step up from 1.1 million in the baseline architecture.

<br>
### Training The Updated Network

I ran the same code to train this updated network as I did for the baseline network (50 epochs) - the only change was that I modified the filename for the saved network to ensure I had all network files for comparison.

<br>
### Analysis Of Training Results

As I again saved my training process to the *history* object, I could analyze & plot the performance (Classification Accuracy, and Loss) of the updated network epoch by epoch.

The below image shows the same two plots I analyzed for the tuned network, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second showing the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

![alt text](/img/posts/cnn-tuned-accuracy-plot.png "CNN Tuned Accuracy Plot")

<br>
Firstly, I could see a peak Classification Accuracy on the validation set of around **98%** which was the highest I had seen from all networks so far, just higher than the 97% I saw for the addition of Image Augmentation to my baseline network.

As Dropout & Image Augmentation was in place here, I again saw the elimination of overfitting.

<br>
### Performance On The Test Set

During training, I assessed my updated network performance on both the training set and the validation set. Here, as I did for the baseline & Dropout networks, I would get a view of how well my network performed when predicting on data that was *no part* of the training process whatsoever - (my test set).

I ran the same code as I did for the earlier networks, with the only change being to ensure I was loading in the network file for the updated network

<br>
### Test Set Classification Accuracy

My optimized network, with both Dropout & Image Augmentation in place, scored **95%** on the Test Set, again marginally higher than what I had seen from the other networks so far.

<br>
### Test Set Confusion Matrix

As mentioned each time, while overall Classification Accuracy was very useful, it could hide what was going on with the network's predictions!

Our 95% Test Set accuracy at an *overall* level told us that I didn't have too much to worry about here, but let's take a look anyway and see if anything interesting poped up.

Running the same code from the baseline section on results for my updated network, I got the following output:

```
actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              0.9      0.0     0.0   0.0    0.0     0.0
avocado            0.0      1.0     0.0   0.0    0.0     0.0
banana             0.0      0.0     0.9   0.0    0.0     0.0
kiwi               0.0      0.0     0.0   0.9    0.0     0.0
lemon              0.0      0.0     0.0   0.0    1.0     0.0
orange             0.0      0.0     0.0   0.1    0.0     1.0
```
<br>
Along the top were my *actual* classes and down the side were my *predicted* classes - so by counting *down* the columns I could get the Classification Accuracy (%) for each class, and I could see where it was getting confused.

So, while overall my test set accuracy was 95% - for each class I saw:

* Apple: 90%
* Avocado: 100%
* Banana: 90%
* Kiwi: 90%
* Lemon: 100%
* Orange: 100%

All classes here were predicted *at least as accurately or better* when compared to the best network so far - so my optimized architecture did appear to have helped!

___
<br>
# Transfer Learning With VGG16 <a name="cnn-transfer-learning"></a>

<br>
#### Transfer Learning Overview

Transfer Learning is an extremely powerful way for us to utilize pre-built, and pre-trained networks, and apply these in a clever way to solve *our* specific Deep Learning based tasks.  It consists of taking features learned on one problem, and leveraging them on a new, similar problem!

For image-based tasks this often means using all the the *pre-learned* features from a large network, so all of the convolutional filter values and feature maps, and instead of using it to predict what the network was originally designed for, piggybacking it, and training just the last part for some other task.

The hope is, that the features which have already been learned will be good enough to differentiate between our new classes, and we’ll save a whole lot of training time (and be able to utilize a network architecture that has potentially already been optimized).

Here, I utilized a famous network known as **VGG16**.  This was designed back in 2014, but even by today's standards is a fair heft network. It was trained on the famous *ImageNet* dataset, with over a million images across one thousand different image classes. Everything from goldfish to cauliflowers to bottles of wine, to scuba divers!

![alt text](/img/posts/vgg16-architecture.png "VGG16 Architecture")

<br>
The VGG16 network won the 2014 ImageNet competition, meaning that it predicted more accurately than any other model on that set of images (although this has now been surpassed).

If I could get my hands on the fully trained VGG16 model object, built to differentiate between all of those one thousand different image classes, the features that were contained in the layer before flattening would be very rich, and could be very useful for predicting all sorts of other images too without having to (a) re-train this entire architecture, which would be computationally, very expensive or (b) having to come up with my very own complex architecture, which I knew could take a lot of trial and error to get right!

All the hard work had been done, I just wanted to "transfer" those "learnings" to my own problem space.

<br>
### Updated Data Pipeline

My data pipeline would remain *mostly* the same as it was when applying my own custom-built networks - but there were some subtle changes. In the code below I needed to import VGG16 and the custom preprocessing logic that it used. I also needed to send my images in with the size 224 x 224 pixels as this was what VGG16 expects. Otherwise, the logic stayed as was.

```python
# Import the required Python libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 224
img_height = 224
num_channels = 3
num_classes = 6

# Image generators
training_generator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5,1.5),
                                        fill_mode = 'nearest')
                                        
validation_generator = ImageDataGenerator(rescale = 1./255)

# Image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                                      target_size = (img_width, img_height),
                                                                      batch_size = batch_size,
                                                                      class_mode = 'categorical')
```

<br>
### Network Architecture

Keras makes the use of VGG16 very easy. I downloaded the *bottom* of the VGG16 network (everything up to the Dense Layers) and added in what I needed to apply the *top* of the model to my fruit classes.

I then specified that I *did not* want the imported layers to be re-trained, I wanted their parameter values to be frozen.

The original VGG16 network architecture contains two massive Dense Layers near the end, each with 4096 neurons. Since my task of classifying 6 types of fruit was more simplistic than the original 1000 ImageNet classes, I reduced this down and instead implemented two Dense Layers with 128 neurons each, followed by my output layer.

```python
# Network architecture
vgg = VGG16(input_shape = (img_width, img_height, num_channels), include_top = False)

# Freeze all layers (they won't be updated during training)
for layer in vgg.layers:
    layer.trainable = False

flatten = Flatten()(vgg.output)

dense1 = Dense(128, activation = 'relu')(flatten)
dense2 = Dense(128, activation = 'relu')(dense1)

output = Dense(num_classes, activation = 'softmax')(dense2)

model = Model(inputs = vgg.inputs, outputs = output)

# Compile network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# View network architecture
model.summary()
```
<br>
The below shows my final architecture:

```
______________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_7 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_14 (Dense)             (None, 128)               3211392   
_________________________________________________________________
dense_15 (Dense)             (None, 128)               16512     
_________________________________________________________________
dense_16 (Dense)             (None, 6)                 774       
=================================================================
Total params: 17,943,366
Trainable params: 3,228,678
Non-trainable params: 14,714,688
```

<br>
My VGG16 architecture had a total of 17.9 million parameters, much bigger than what I had built so far. Of this, 14.7 million parameters were frozen, and 3.2 million parameters would be updated during each iteration of back-propagation, and these were going to be figuring out exactly how to use those frozen parameters that were learned from the ImageNet dataset, to predict my classes of fruit!

<br>
### Training The Network

I ran the same code to train this updated network as I did for the baseline network, although to start with for only 10 epochs as it was a much more computationally expensive training process.

<br>
### Analysis Of Training Results

As I saved my training process to the *history* object, I could analyze & plot the performance (Classification Accuracy, and Loss) of the updated network epoch by epoch.

The below image shows the same two plots I analyzed for the tuned network, the first showing the epoch by epoch **Loss** for both the training set (blue) and the validation set (orange) & the second showing the epoch by epoch **Classification Accuracy** again, for both the training set (blue) and the validation set (orange).

![alt text](/img/posts/cnn-vgg16-accuracy-plot.png "VGG16 Accuracy Plot")

<br>
Firstly, I could see a peak Classification Accuracy on the validation set of around **98%** which was equal to the highest I had seen from all networks so far, but what was impressive was that it achieved this in only 10 epochs!

<br>
### Performance On The Test Set

During training, I assessed my updated network performance on both the training set and the validation set.  Here, as I did for all other networks, I would get a view of how well my network performs when predicting data that was *no part* of the training process whatsoever - (my test set).

I ran the same code as I did for the earlier networks, with the only change being to ensure I was loading in the network file for the updated network

<br>
### Test Set Classification Accuracy

My VGG16 network scored **98%** on the Test Set, higher than that of my best custom network.

<br>
### Test Set Confusion Matrix

As mentioned each time, while overall Classification Accuracy was very useful, it could hide what was going on with the network's predictions!

My 98% Test Set accuracy at an *overall* level tells me that I didn't have too much to worry about here, but for comparison's sake let's take a look!

Running the same code from the baseline section on results for my updated network, I got the following output:

```
actual_label     apple  avocado  banana  kiwi  lemon  orange
predicted_label                                             
apple              1.0      0.0     0.0   0.0    0.0     0.0
avocado            0.0      1.0     0.0   0.0    0.0     0.0
banana             0.0      0.0     1.0   0.0    0.0     0.0
kiwi               0.0      0.0     0.0   1.0    0.0     0.0
lemon              0.0      0.0     0.0   0.0    0.9     0.0
orange             0.0      0.0     0.0   0.0    0.1     1.0
```
<br>
Along the top were my *actual* classes and down the side were my *predicted* classes - so by counting *down* the columns I could get the Classification Accuracy (%) for each class, and I could see where it was getting confused.

So, while overall my test set accuracy was 98% - for each class I saw:

* Apple: 100%
* Avocado: 100%
* Banana: 100%
* Kiwi: 100%
* Lemon: 90%
* Orange: 100%

All classes here were predicted *at least as accurate or better* when compared to the best custom network!

___
<br>
# Overall Results Discussion <a name="cnn-results"></a>

I made some huge strides in terms of making my network's predictions more accurate, and more reliable on new data.

My baseline network suffered badly from overfitting - the addition of both Dropout & Image Augmentation eliminated this almost entirely.

In terms of Classification Accuracy on the Test Set, I saw:

* Baseline Network: **75%**
* Baseline + Dropout: **85%**
* Baseline + Image Augmentation: **93%**
* Optimised Architecture + Dropout + Image Augmentation: **95%**
* Transfer Learning Using VGG16: **98%**

Tuning the network architecture with Keras-Tuner gave me a great boost, but was also very time-intensive - however if this time investment results in improved accuracy then it is time well spent.

The use of Transfer Learning with the VGG16 architecture was also a great success, in only 10 epochs I was able to beat the performance of my smaller, custom networks which was training over 50 epochs. From a business point of view, I also needed to consider the overheads of (a) storing the much larger VGG16 network file, and (b) any increased latency on inference.

___
<br>
# Growth & Next Steps <a name="growth-next-steps"></a>

The proof of concept was successful, I had shown that I could get very accurate predictions albeit in a small number of classes. I needed to showcase this to the client, discuss what it was that made the network more robust, and then looked to test my best networks on a larger array of classes.
