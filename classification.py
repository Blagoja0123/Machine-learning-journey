from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

CSV_COL_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COL_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COL_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

print(train.head())

def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

feature_cols = []

for key in train.keys():
    feature_cols.append(tf.feature_column.numeric_column(key=key))

print(feature_cols)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_cols,
    hidden_units=[30,10],
    n_classes=3
)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000
)

eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

def test_input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid: 
    val = input(feature + ": ")
    if not val.isdigit(): valid = False

  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: test_input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))