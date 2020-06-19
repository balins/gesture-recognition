import tensorflow as tf
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

suspend_on_finished = True
model_name = '0.96_0.96_0.93_0.99_1591195107.h5'
image_size = 150

params_generator = {
    'rescale': 1. / 255,
    'horizontal_flip': True,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.3,
    'rotation_range': 45
}

params_flow = {
    'target_size': (image_size, image_size),
    'batch_size': 128,
    'color_mode': 'grayscale',
    'class_mode': 'categorical',
    'seed': 42
}

train_datagen = ImageDataGenerator(**params_generator)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    **params_flow
)

val_datagen = ImageDataGenerator(**params_generator)
validation_generator = val_datagen.flow_from_directory(
    'data/val',
    **params_flow
)

test_datagen = ImageDataGenerator(**params_generator)
test_generator = test_datagen.flow_from_directory(
    'data/test',
    **params_flow
)

if model_name:
    model = load_model('baseline_models/{0}'.format(model_name))
else:
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(image_size, image_size, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print(train_generator.class_indices, end='\n\n')

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

model.fit_generator(train_generator, epochs=7, verbose=2,
                    callbacks=[early_stopping], validation_data=validation_generator)

_, acc = model.evaluate_generator(test_generator)
print('Test Accuracy: %.3f' % (acc * 100))

if model_name:
    model.save('models/%.2f_%s' % (acc, model_name))
else:
    import time

    ts = time.gmtime()
    simple_ts = time.strftime("%s", ts)
    model.save('models/%.2f_%s.h5' % (acc, simple_ts))

if suspend_on_finished:
    import os

    os.system("systemctl suspend")