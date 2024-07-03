import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Function to load images and labels from a directory
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize images to 64x64
            images.append(img)
            # Assuming dog images start with 'dog' and cat images start with 'cat'
            if 'dog' in filename:
                labels.append('dog')
            elif 'cat' in filename:
                labels.append('cat')
    return images, labels

# Load training data
train_folder = r'D:\Prodigy\Task-3\train'
train_images, train_labels = load_images_from_folder(train_folder)

# Convert lists to numpy arrays
X = np.array(train_images)
y = np.array(train_labels)

# Flatten the images for SVM input
X = X.reshape(X.shape[0], -1)

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = svm_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy}')
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# Load test data
test_folder = r'D:\Prodigy\Task-3\test1'
test_images, test_labels = load_images_from_folder(test_folder)  # If test labels are not available, skip labels part

# Convert lists to numpy arrays
X_test = np.array(test_images)

# Flatten the images for SVM input
X_test = X_test.reshape(X_test.shape[0], -1)

# Predict on the test set
y_test_pred = svm_model.predict(X_test)

# Decode the predictions
y_test_pred_labels = le.inverse_transform(y_test_pred)

# Save the predictions to a file
with open('test_predictions.csv', 'w') as f:
    f.write('Filename,Label\n')
    for filename, label in zip(os.listdir(test_folder), y_test_pred_labels):
        f.write(f'{filename},{label}\n')

print('Test predictions saved to test_predictions.csv')
