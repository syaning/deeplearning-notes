import keras
from keras.layers import (
    Input,
    ZeroPadding2D,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Flatten,
    Dense
)
from keras.models import Model
from kt_utils import *


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(shape=input_shape)
    X = ZeroPadding2D(padding=(1, 1))(X_input)
    X = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)

    X = Flatten()(X)
    Y = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=Y, name='HappyModel')

    return model


if __name__ == '__main__':
	# 1. Load dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    
    # 2. Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # 3. Reshape
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    # 4. Inspect data
    print('number of training examples = ' + str(X_train.shape[0]))
    print('number of test examples = ' + str(X_test.shape[0]))
    print('X_train shape: ' + str(X_train.shape))
    print('Y_train shape: ' + str(Y_train.shape))
    print('X_test shape: ' + str(X_test.shape))
    print('Y_test shape: ' + str(Y_test.shape))

    # 5. Train
    happyModel = HappyModel((64, 64, 3))
    happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                       loss='binary_crossentropy', metrics=['accuracy'])
    happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)

    # 6. Evaluate
    preds = happyModel.evaluate(x=X_test, y=Y_test)
    print('Loss = ' + str(preds[0]))
    print('Test Accuracy = ' + str(preds[1]))
