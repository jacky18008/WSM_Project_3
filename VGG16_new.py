from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Dense

num_classes = 1


#build model
model = VGG16(weights='imagenet', include_top=False)

model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []

x=Dense(num_classes, activation='sigmoid')(model.output)

model=Model(model.input,x)

model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = model.predict(img)


