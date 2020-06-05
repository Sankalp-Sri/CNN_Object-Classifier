
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import os 
os.getcwd()

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 64, activation = 'relu'))

classifier.add(Dense(units = 5, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.summary()

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range=45)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('E:/Practice Projects/face_classifier/Images/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 )

test_set = test_datagen.flow_from_directory('E:/Practice Projects/face_classifier/Images/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            )

superhero_classifier = classifier.fit_generator(training_set,
                         steps_per_epoch = 5000,
                         epochs = 1,
                         validation_data = test_set,    
                         validation_steps = 2000)

classifier.save("superhero_classifier.h5")
print("Saved model to disk")

# Part 3 - Making new predictions




import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model = load_model('superhero_classifier.h5')

test_image = image.load_img('E:/Practice Projects/face_classifier/Images/train/Iron Man/2732f2d35f55e96b0a597df76f6b9b85.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)


result = model.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'Captain America'
    print(prediction)
elif result[0][1]==1:
    prediction = 'Hulk'
    print(prediction)
elif result[0][2]==1:
    prediction = 'Iron Man'
    print(prediction)
elif result[0][3]==1:
    prediction = 'Spider Man'
    print(prediction)
elif result[0][4]==1:
    prediction = 'Thor'
    print(prediction)