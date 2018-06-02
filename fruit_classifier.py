

# Importing the Keras libraries and packages
from gtts import gTTS
import os
from pygame import mixer 
import time
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))

# doupout
classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim = 6, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

'''test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')'''

classifier.fit_generator(training_set,
                         samples_per_epoch = 119,
                         nb_epoch = 50,
                         )

import numpy as np 
from keras.preprocessing import image
test_image =image.load_img('14.jpeg', target_size =(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis= 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result)

if result[0][0]==1:
	#prediction ='apple'
  mytext = 'apple'
elif result[0][1]:
	#prediction='banana'
  mytext = 'banana'
elif result[0][2]:
  #prediction ='grapes'
  mytext = 'grapes'
elif result[0][3]:
  #prediction ='kiwi'
  mytext = 'kiwi'
elif result[0][4]:
  #prediction ='mango'
  mytext = 'mango'
elif result[0][5]:
  #prediction ='orange'
  mytext = 'orange'
else:
  #prediction ="hello"
  mytext = 'no answer'

#print(prediction)

language = 'en'
 

myobj = gTTS(text=mytext, lang=language, slow=False)
 
# Saving the converted audio in a mp3 file named
# welcome 
myobj.save("welcome1.mp3")
 
# Playing the converted file
from pygame import mixer # Load the required library
import time
mixer.init()
mixer.music.load('/home/rajesh/Desktop/fruit/welcome1.mp3')
mixer.music.play()
time.sleep(10)

