# Fashion MNIST CNN Classifier 

A simple convolutional neural network (CNN) built using TensorFlow/Keras to classify Fashion MNIST images into 10 clothing categories.

## Model Highlights
- Achieved **91.48% accuracy** on test data
- Input shape: 28x28 grayscale images
- Used 3 convolutional layers with dropout and max-pooling
- Trained on 60,000 images, validated on 6,000, tested on 10,000

**Classes:**

0 = T-shirt/top  
1 = Trouser  
2 = Pullover  
3 = Dress  
4 = Coat  
5 = Sandal  
6 = Shirt  
7 = Sneaker  
8 = Bag  
9 = Ankle boot


**Folder Structuer**
```
Fashion_MNIST_CNN/
├── fashion_image_classification.ipynb           ← Jupyter Notebook
├── fashion_x_train.npy         ← Dataset files
├── fashion_y_train.npy
├── fashion_x_test.npy
├── fashion_y_test.npy
```
