import os
import cv2
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
import matplotlib.pyplot as plt

# Paths to the dataset files
data_path = 'pec/'
images_path = os.path.join(data_path, 'images/')
meta_path = os.path.join(data_path, 'meta/')

# Load classes and labels
with open(os.path.join(meta_path, 'classes.txt'), 'r') as f:
    classes = f.read().strip().split('\n')
class_dict = {cls: idx for idx, cls in enumerate(classes)}

with open(os.path.join(meta_path, 'labels.txt'), 'r') as f:
    labels = f.read().strip().split('\n')
calories = {line.split()[0]: float(line.split()[1]) for line in labels}

# Load data from train.json
with open(os.path.join(meta_path, 'train.json'), 'r') as f:
    train_data = json.load(f)


# Load and preprocess images and labels
def load_images_and_labels(train_data, images_path, class_dict, calories):
    X = []
    y_class = []
    y_calories = []
    for item in train_data:
        image_path = os.path.join(images_path, item['class'], f"{item['image_id']}.jpg")
        if os.path.exists(image_path):
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize to 128x128
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                X.append(img.flatten())  # Flatten for SVM input
                y_class.append(class_dict[item['class']])
                y_calories.append(calories[item['image_id']])

    X = np.array(X) / 255.0  # Normalize
    y_class = np.array(y_class)
    y_calories = np.array(y_calories)

    return X, y_class, y_calories


X, y_class, y_calories = load_images_and_labels(train_data, images_path, class_dict, calories)

# Split data into training and test sets
X_train, X_test, y_class_train, y_class_test, y_calories_train, y_calories_test = train_test_split(
    X, y_class, y_calories, test_size=0.3, random_state=42
)
# Classifier for food item recognition
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_class_train)

# Regressor for calorie estimation
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_calories_train)
# Predict on the test set for classification
y_class_pred = clf.predict(X_test)
print(f'Classification Accuracy: {accuracy_score(y_class_test, y_class_pred)}')
print(classification_report(y_class_test, y_class_pred, target_names=classes))

# Predict on the test set for regression
y_calories_pred = regressor.predict(X_test)
print(f'Mean Absolute Error for Calorie Estimation: {mean_absolute_error(y_calories_test, y_calories_pred)}')

# Visualize some predictions
for i in range(5):
    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[i].reshape(128, 128, 3))
    plt.title(f'Test Image {i}')

    # Prediction
    class_name = classes[y_class_pred[i]]
    actual_calories = y_calories_test[i]
    predicted_calories = y_calories_pred[i]
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f'Class: {class_name}\nActual Calories: {actual_calories:.2f}\nPredicted Calories: {predicted_calories:.2f}',
             fontsize=12, va='center', ha='center')
    plt.title(f'Prediction for Image {i}')
    plt.show()
# Load and preprocess test images from test.txt
with open(os.path.join(meta_path, 'test.txt'), 'r') as f:
    test_files = f.read().strip().split('\n')

test_images = []
test_image_ids = []
for file in test_files:
    class_name, image_id = file.split()
    image_path = os.path.join(images_path, class_name, f"{image_id}.jpg")
    if os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to 128x128
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            test_images.append(img.flatten())  # Flatten for SVM input
            test_image_ids.append((class_name, image_id))

test_images = np.array(test_images) / 255.0  # Normalize

# Make predictions
test_class_preds = clf.predict(test_images)
test_calorie_preds = regressor.predict(test_images)

# Save predictions to a CSV file
with open('test_predictions.csv', 'w') as f:
    f.write('Class,ImageID,PredictedCalories\n')
    for (class_name, image_id), pred_class, pred_calorie in zip(test_image_ids, test_class_preds, test_calorie_preds):
        f.write(f'{class_name},{image_id},{pred_calorie:.2f}\n')

print('Test predictions saved to test_predictions.csv')
