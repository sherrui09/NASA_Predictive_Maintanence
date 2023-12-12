# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

# Keras imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# run lstm_data_processing.py here

# function to generate labels that align with sequences
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements]

# generate labels for multi-class classification
label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, 'label2')
             for id in train_df['id'].unique()]
label_array = np.concatenate(label_gen).astype(np.int)

# Splitting the data into train and test sets for multi-class classification
x_train, x_test, y_train, y_test = train_test_split(seq_array, label_array, test_size=0.2, random_state=42)


# Update number of classes based on label2 distribution
nb_classes = 3  # label2 has 3 classes: 0, 1, and 2


# One-hot encode the labels
y_train_one_hot = to_categorical(y_train, num_classes=nb_classes)
y_test_one_hot = to_categorical(y_test, num_classes=nb_classes)


# Recompute class weights for the new label distribution
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
class_weight_dict = {i: class_weights[i] for i in range(nb_classes)}

print("Shape of x_train:", x_train.shape)
print("Shape of y_train_one_hot:", y_train_one_hot.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test_one_hot:", y_test_one_hot.shape)

# Check class distribution in one-hot encoded labels
train_class_distribution = np.sum(y_train_one_hot, axis=0)
test_class_distribution = np.sum(y_test_one_hot, axis=0)

print("Class distribution in y_train_one_hot:", train_class_distribution)
print("Class distribution in y_test_one_hot:", test_class_distribution)

# multi-class lstm architecture
def build_lstm_multi_class_model(input_shape, output_units):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_units, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Parameters
input_shape = (x_train.shape[1], x_train.shape[2])
output_units = 3
# Rebuild the model
model = build_lstm_multi_class_model(input_shape, output_units)

# Retrain with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
history = model.fit(x_train, y_train_one_hot, epochs=50, batch_size=64, validation_split=0.1, callbacks=[early_stop])

# Generate predictions
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_classes)
print("Confusion Matrix:\n", conf_matrix)

# Generate classification report
class_report = classification_report(y_test, predicted_classes)
print("\nClassification Report:\n", class_report)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Predict probabilities
y_pred_prob = model.predict(x_test)

# Plot ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_one_hot.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.show()