import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_image_model():
    train_datagen = ImageDataGenerator(
        rescale = 1./255 ,
        shear_range = 0.2 ,
        zoom_range = 0.2 ,
        horizontal_flip= True
    )
    training_set = train_datagen.flow_from_directory(
        'datasets/casia_images/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )

    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(
        'datasets/casia_images/validation',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )

    model = Sequential()
    model.add(Conv2D(filters = 32 , kernel_size = 3 ,strides=(1,1), activation = 'relu',input_shape= [128 , 128 ,3]))
    model.add(MaxPooling2D(pool_size = 2 ,strides = 2))
    model.add(Conv2D(filters = 32 , kernel_size = 3 ,strides=(1,1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2 ,strides = 2))
    model.add(Flatten())
    model.add(Dense(units=128 , activation='relu'))
    model.add(Dense(units=1 , activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(training_set, validation_data=test_set, epochs=10)

    model.save("models/image_cnn_model.h5")

if __name__ == "__main__":
    train_image_model()
