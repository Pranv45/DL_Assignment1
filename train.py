import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist 

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Find one example per class
sample_images = []
sample_labels = []
for class_idx in range(10):
    index = np.where(y_train == class_idx)[0][0]  # Get the first occurrence of each class
    sample_images.append(x_train[index])
    sample_labels.append(class_names[class_idx])

# Plot images in a grid
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Fashion-MNIST Sample Images", fontsize=14)

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i], cmap='gray')
    ax.set_title(sample_labels[i])
    ax.axis('off')

plt.show()