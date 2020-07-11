from tensorflow.keras import datasets, layers, models
import tensorflow.keras as keras
import numpy as np

(train_images, train_labels_raw), (test_images, test_labels_raw) = datasets.mnist.load_data()

train_images = train_images / 255
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images / 255
test_images = test_images.reshape(-1, 28, 28, 1)

train_labels = []
for label in train_labels_raw:
    output = [0 for x in range(10)]
    output[label] = 1
    train_labels.append(output)
train_labels = np.array(train_labels)

test_labels = []
for label in test_labels_raw:
    output = [0 for x in range(10)]
    output[label] = 1
    test_labels.append(output)
test_labels = np.array(test_labels)


model = models.Sequential()
model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# print(model.summary())

model.compile(optimizer='Nadam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=3, validation_data=(test_images, test_labels))

#model.save('DC')
model.save('DCTF.h5')

model.save_weights('DC.h5')
with open('DC.json', 'w') as json_out:
    json_out.write(model.to_json())
