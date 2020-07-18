import sys
from tensorflow.keras.metrics import CategoricalAccuracy
import numpy as np

cat_accuracy = CategoricalAccuracy()

with open(sys.argv[1], 'r') as fin:
    labels = [[float(x) for x in line.split(' ') if len(x) > 0] for line in fin.read().split('\n') if len(line) > 0]

with open(sys.argv[2], 'r') as fin:
    predictions = [[float(x) for x in line.split(' ') if len(x) > 0] for line in fin.read().split('\n') if len(line) > 0]

cat_accuracy.update_state(labels, predictions)
print(f'Categorical Accuracy: {cat_accuracy.result().numpy():.4f}')
