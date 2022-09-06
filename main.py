import pandas as pd
import tensorflow as tf

from vae.basic_vae import BasicVAE
from sklearn.metrics import precision_recall_curve
import numpy as np

df = pd.read_csv("./sample_data.csv")
test_portion = 0.3
test_n = int(df.shape[0] * test_portion)
train_values, test_values = df['value'].values[:-test_n], df['value'].values[-test_n:]
train_labels, test_labels = df['label'].values[:-test_n], df['label'].values[-test_n:]

# window for train without anomalies
train_value_windows = []
for i in range(len(train_values)-119):
    train_value_windows.append(train_values[i: i + 120])

# test windows
test_value_windows = []
for i in range(test_n-119):
    test_value_windows.append(test_values[i: i + 120])

print('The size of train_value_windows is {}.'.format(len(train_value_windows)))
print('The size of test_value_windows is {}.'.format(len(test_value_windows)))

vae_model = BasicVAE()
vae_model.fit(train_value_windows, batch_size=1, train_epochs=2)

test_value_windows = tf.cast(test_value_windows, tf.float32)
test_batches = tf.data.Dataset.from_tensor_slices(test_value_windows).batch(1)

reconstructions = [vae_model.reconstruction_prob(window, 10) for window in test_batches]
precisions, recalls, thresholds = precision_recall_curve(test_labels[-len(reconstructions):], reconstructions)

# get best f1-score and best threshold
f1_scores = (2 * precisions * recalls) / (precisions + recalls)
best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
best_thresholds = thresholds[np.argmax(f1_scores[np.isfinite(f1_scores)])]

print('The best threshold: {:.4f}'.format(best_thresholds))
print('The best f1-score: {:.4f}'.format(best_f1_score))
