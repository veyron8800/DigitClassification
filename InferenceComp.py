from tensorflow.keras import datasets
from tensorflow import keras
import numpy as np
import hls4ml
import yaml

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
tf_predictions = model.predict(test_images)

with open('DC.yml', 'r') as fin:
    cfg = yaml.load(fin)

hls_model = hls4ml.converters.keras_to_hls(cfg)
hls_model.compile()
hls_predictions = hls_model.predict(test_images)

tf_out = open('InferenceComp/TF_OUT.dat', 'w')
hls_out = open('InferenceComp/HLS4ML_Out_Master_Adj_Type.dat', 'w')
labels_out = open('InferenceComp/Labels_Out.dat', 'w')

for tf_pred, hls_pred, labels in zip(tf_predictions, hls_predictions, test_labels):
    tf_out.write(' '.join(tf_pred.astype(str)) + '\n')
    hls_out.write(' '.join(hls_pred.astype(str)) + '\n')
    labels_out.write(' '.join(labels.astype(str)) + '\n')

tf_out.close()
hls_out.close()
labels_out.close()
