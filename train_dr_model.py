import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from matplotlib import pyplot as plt

def load_dataset():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 60000 images of 28 by 28 grayscale numbers
    print(x_train.shape, y_train.shape)

    # plot first few images
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()

    # As input, a CNN takes tensors of shape (image_height, image_width, color_channels)
    # add an extra 1 dimension for colour (grayscale)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # 10 digits
    num_classes = 10
    # convert class vectors to binary class matrices
    # uses a matrix to store digit labels instead of an array where col = classification and row = input
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # normalize grayscale values from 0 to 1
    x_train /= 255.0
    x_test /= 255.0

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return x_train, x_test, y_train, y_test

def define_model():
    input_shape = (28, 28, 1)
    num_classes = 10
    model = Sequential()
    # convolution is a linear operation, at a basic level filtering for interesting features through multiplication
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_uniform',input_shape=input_shape))
    # max pooling takes the maximum in a 2x2 grid with strides of 2, effectively down sampling by factor of 2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # more feature extraction layers
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    # dense layer to interpret features
    model.add(Dense(256, activation='relu',kernel_initializer='he_uniform'))
    # softmax results in a probability distribution of size num_classes
    model.add(Dense(num_classes, activation='softmax'))
    # gradient descent
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])
    return model

def evaluate_model(x_train, x_test, y_train, y_test, model):
    scores, histories = list(), list()
    batch_size = 128
    num_classes = 10
    epochs = 10
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    print("The model has successfully trained")
    model.save('mnist.h5')
    print("Saving the model as mnist.h5")

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    scores.append(score)
    histories.append(hist)
    return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

# entry point
x_train, x_test, y_train, y_test = load_dataset()
model = define_model()
scores, histories = evaluate_model(x_train, x_test, y_train, y_test, model)
summarize_diagnostics(histories)
