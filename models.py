from __future__ import print_function
import warnings
warnings.simplefilter('ignore')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier


def trainModel(input_shape
               ,xTrain, yTrain
               ,xTest, yTest
               ,ageMatchUnmatch
               ,dataset
               ,num_classes):

    batch_size = 64
    epochs = 64

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(12, activation='relu', name="dense_one"))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name="dense_two"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(xTrain, yTrain,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(xTest, yTest))
    model.save('Models/'+ ageMatchUnmatch+ "_"+ dataset+'.h5')


    trainScore = model.evaluate(xTrain, yTrain, verbose=0)
    testScore  = model.evaluate(xTest,  yTest,  verbose=0)


    print('Train loss: ', trainScore[0], '           Train accuracy: ', trainScore[1])
    print('Test loss:  ', testScore[0],  '           Test accuracy:  ', testScore[1])

    