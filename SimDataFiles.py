from tensorflow.keras import datasets
from tensorflow import keras
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

model = keras.models.load_model('DCTF.h5')
predictions = model.predict(test_images)

fout_images_flat_c = open('SimData/input_c.dat', 'w')
fout_images_flat_f = open('SimData/input_f.dat', 'w')
fout_labels = open('SimData/labels.dat', 'w')
fout_model_prediction = open('SimData/ModelPredictions.dat', 'w')

for i in range(100):
    fout_labels.write(' '.join(test_labels[i].astype(str)) + '\n')
    fout_images_flat_c.write(' '.join(test_images[i].flatten('C').astype(str)) + '\n')
    fout_images_flat_f.write(' '.join(test_images[i].flatten('F').astype(str)) + '\n')
    fout_model_prediction.write(' '.join(predictions[i].astype(str)) + '\n')

fout_images_flat_c.close()
fout_images_flat_f.close()
fout_labels.close()
fout_model_prediction.close()
