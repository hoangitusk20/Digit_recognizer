# Digit Recognizer 
This project aims to practice and solidify my understanding of what I have learned during my deep learning course. By implementing the "Digit Recognizer," I have gained a better grasp of the concepts related to deep learning and how to use PyTorch to build neural network models.

## Problem Statement

Given an image, the goal is to predict the digit it represents. For instance, given the image below, the model should predict the digit '2'.

![image.png](image.png)

## Solution Approach

### Solution 1: Use logistic regression
Before diving into deep learning, I had learned about machine learning. Therefore, before creating a neural network to solve this problem, I will use Logistic Regression for an initial prediction. Then, I'll compare the results with the deep learning model.

#### Tools and Libraries Used
- Language: Python (Jupyter Notebook)
- Libraires: numpy, pandas, Scikit-learn

#### Summary
By using Scikit-learn, I quickly built a model, trained, and made predictions using readily available functions within this library. The result achieved an accuracy of 92.6% on the validation set.

### Solution 2: Build one hidden layer neural network from scratch
The purpose of building this model is to gain a deeper understanding of how a neural network operates. After completing the construction, I have a clear understanding of what each step in this model does.
#### Tools and Libraries Used
- Language: Python (Jupyter Notebook)
- Libraries: NumPy, Pandas
#### Neural Network Architecture
I built a neural network with a single hidden layer:
- Input: $X$ with dimensions $(\text{nPixel} \times m)$, where $\text{nPixel}$ is the number of pixels in the image, and $m$ is the number of images in the training set.
- Hidden Layer: $nUnits$ neurons with the $ReLU$ activation function.
- Output Layer: 10 neurons representing digits 0 to 9, with the softmax activation function.

##### Forward Propagation

The forward propagation process can be described as follows:

1. $Z_1 = W_1 \cdot X + b_1$
2. $A_1 = \text{ReLU}(Z_1)$
3. $Z_2 = W_2 \cdot A_1 + b_2$
4. $A_2 = \text{Softmax}(Z_2)$

##### Backward Propagation

The backward propagation process involves computing gradients and updating parameters:

1. $dz_2 = A_2 - Y$
2. $dw_2 = \frac{1}{m} \cdot dz_2 \cdot A_1^T$
3. $db_2 = \frac{1}{m} \cdot \text{sum}(dz_2, \text{axis}=1, \text{keepdims=True})$
4. $dz_1 = W_2^T \cdot dz_2 \cdot (\text{ReLU}'(Z_1))$
5. $dw_1 = \frac{1}{m} \cdot dz_1 \cdot X^T$
6. $db_1 = \frac{1}{m} \cdot \text{sum}(dz_1, \text{axis}=1, \text{keepdims=True})$

##### Result
Achieved an accuracy of 99% on validation set and 98.942% on test set



##### Summary:
Although it's a very simple model with just one hidden layer, this model still produces quite good results and better than the Logistic Regression approach. This model achieved an accuracy of 95.7% on validation set.

### Solution 3: Build multi hidden layer neural network using Pytorch
PyTorch is a highly popular framework used for building deep learning models. The purpose of this solution is for me to practice how to construct models using PyTorch.

#### Tools and Libraries Used
- Language: Python (Jupyter Notebook)
- Libraries: NumPy, Pandas, Pytorch.

#### The construction process
- Read data.
- Use DataLoader to divide the data into mini-batches (optional).
- Build model class. , which includes two main functions: \___init()\___ containing functions used in the forward pass and the forward() function to call these functions in the correct order of the forward propagation process.
- Choose an appropriate optimizer and loss function.
- Training loop:
  + Prediction (forward pass).
  + Calculate the loss.
  + Reset the gradients of the parameters to zero.
  + Compute gradients.
  + Update the parameters.
- Evaluate the model.
#### Summary
By using available functions, building a model with PyTorch becomes much easier. I constructed a model with 3 hidden layers effortlessly and achieved an accuracy of 97% on the validation set. Furthermore, PyTorch supports GPU computing, which significantly speeds up the execution of my model.

### Solution 4: Build convolutional neural network using Pytorch
This approach is the one that achieved the highest score among the options I submitted on Kaggle. By building a convolutional neural network model using PyTorch, I gained a better understanding of the effectiveness of CNNs in image-related predictions.

#### Tools and Libraries Used
- Language: Python (Jupyter Notebook)
- Libraries: NumPy, Pandas, Pytorch.

#### Summary
CNNs are specially created to handle image data, which is why their performance is a bit better than traditional neural networks. Convolutional layers are better at learning image features compared to traditional neural networks, which treat each pixel as a independent input. Building a CNN model in PyTorch is quite similar to constructing a traditional neural network. However, instead of creating fully connected layers, CNNs utilize convolutional and pooling layers. Fully connected layers are typically placed at the end of the model to make predictions.

## Key Learnings from This Exercise:

1. A neural network involves two main processes: forward propagation and backward propagation. Initially, weights are randomly initialized. Through matrix operations, information is passed through the network layers until it reaches the output layer. This process is called forward propagation. After reaching the output, error computation is performed. Then, during backward propagation, the derivatives of the weights are calculated. The model's parameters are updated by subtracting the derivatives multiplied by the learning rate. These two processes continue iteratively until the error falls within an acceptable range or a specific number of iterations is completed. To make predictions, we carry out forward propagation using the learned parameters.

2. The number of neurons significantly affects the model's accuracy. Initially, I built a model with one hidden layer and 10 neurons in it, achieving an accuracy of around 0.85. Despite adjusting the learning rate and the number of iterations, the results always in the range of 0.85-0.87. I then decided to increase the number of neurons to 32, 64, and 128. The accuracy improved to over 0.93. In the final attempt, I chose 128 neurons, a learning rate of 0.9, and 500 iterations, achieving an accuracy of 95.7%.

3. Normalizing data for each layer is crucial. In my initial attempt without normalization, the accuracy is only 9.7%. Until I added data normalization, the model's performance significantly improved.

4. PyTorch is indeed a fantastic tool for building neural network models, making the process of constructing neural networks much faster and more convenient.

5. For tasks involving image predictions, it's best to use CNNs. This is a really great method that provides high efficiency.