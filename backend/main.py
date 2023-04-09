import os
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers

damaged_count = 0
undamaged_count = 0

# Define the directories where the train and test data is stored
train_dir = '../data/train'
test_dir = '../data/test'

# Define the image size
img_size = 128

# Function to load images and labels from a directory
def load_data(directory):
    images = []
    labels = []
    for folder in os.listdir(directory):
        if folder == 'damage':
            label = 1
        elif folder == 'no_damage':
            label = 0
        else:
            continue
        for filename in os.listdir(os.path.join(directory, folder)):
            img = cv2.imread(os.path.join(directory, folder, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(label)
    images = np.array(images).reshape(-1, img_size, img_size, 1)
    return images, labels

# Load the train and test data
train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)

# Normalize the pixel values of the images
scaler = MinMaxScaler()
train_images = scaler.fit_transform(train_images.reshape(-1, img_size*img_size))
test_images = scaler.transform(test_images.reshape(-1, img_size*img_size))
train_images = train_images.reshape(-1, img_size, img_size, 1)
test_images = test_images.reshape(-1, img_size, img_size, 1)

# Split the train data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Save the preprocessed images and labels as numpy arrays

val_images = np.array(val_images)
val_labels = np.array(val_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

model = keras.models.load_model('../save')
relPathOfInputImage = "./hurricane.png"
relPathOfOutputImage = "out.png"

# Load image
image = cv2.imread(relPathOfInputImage)
img_size = 128


# Define window size and stride
window_size = (128, 128)
stride = 64

# Generate windows
windows = []
for y in range(0, image.shape[0] - window_size[1] + 1, stride):
    for x in range(0, image.shape[1] - window_size[0] + 1, stride):
        window = image[y:y+window_size[1], x:x+window_size[0]]
        windows.append(window)

# Predict on windows
resized_windows = []
for window in windows:
    resized_window = cv2.resize(window, window_size)
    resized_window = cv2.cvtColor(resized_window, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values of the window
    resized_window = scaler.transform(resized_window.reshape(-1, img_size*img_size)).reshape(img_size, img_size, 1)
    resized_windows.append(resized_window)

resized_windows = np.array(resized_windows)
predictions = model.predict(resized_windows)
predictions = np.squeeze(predictions)

# Define colors for boxes
green_color = (0, 255, 0)
red_color = (0, 0, 255)

# Filter predictions above a certain threshold
threshold = 0.5
damaged_indices = np.where(predictions > threshold)[0]


# Draw boxes on image
for i, index in enumerate(range(0, len(windows))):
    y = (index // ((image.shape[1] - window_size[0]) // stride)) * stride
    x = (index % ((image.shape[1] - window_size[0]) // stride)) * stride
    color = red_color if i in damaged_indices else green_color
    if i in damaged_indices:
        damaged_count +=1
    else:
        undamaged_count += 1
    cv2.rectangle(image, (x, y), (x+window_size[0], y+window_size[1]), color, thickness=2)

# Save image
cv2.imwrite(relPathOfOutputImage, image)

print(f'Number of Undamaged Structures: {undamaged_count}')
print(f'Number of Damaged Structures: {damaged_count}')

