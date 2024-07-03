import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def load_images_from_folders(base_path):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)
        if os.path.isdir(subject_path):
            for gesture in os.listdir(subject_path):
                gesture_path = os.path.join(subject_path, gesture)
                if os.path.isdir(gesture_path):
                    if gesture not in label_dict:
                        label_dict[gesture] = current_label
                        current_label += 1
                    for filename in os.listdir(gesture_path):
                        img_path = os.path.join(gesture_path, filename)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (64, 64))  # Resize images to 64x64
                            images.append(img)
                            labels.append(label_dict[gesture])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, label_dict


# Load dataset
base_path = r'D:\Prodigy\Task-4\leapGestRecog'  # Replace with the actual path to the dataset
images, labels, label_dict = load_images_from_folders(base_path)

# Normalize images
images = images / 255.0

# Flatten the images for SVM input
X = images.reshape(images.shape[0], -1)

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(labels)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
# Predict on the validation set
y_val_pred = svm_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy}')
# Predict on the test set
y_test_pred = svm_model.predict(X_test)

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy}')


# Visualize some predictions
for i in range(10):
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    plt.title(f'True: {le.inverse_transform([y_test[i]])[0]}, Pred: {le.inverse_transform([y_test_pred[i]])[0]}')
    plt.show()


