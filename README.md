# Fully-Connected-Neural-Nets-from-scratch

**Aim:** <br>

Use the MNIST data set and implement the forward and backward passes for fully-connected deep neural networks from scratch.

**Model Explanation:** <br>

First, let us preprocess the data. We load the training and testing data from the MNIST dataset, consisting of 60,000 training images + labels, and 10,000 testing images + labels. Next, we normalize the image data by dividing each pixel by 255.0 and reshape the labels. Now, we encode the class value of the labels into one-hot vectors to avoid bias implied by the label value.

Now, to build the model, we define a class named as Network which initializes the number of layers, number of nodes in a layer, weights, and biases of the network. Next, we define the forward and backward pass of backpropagation together in a function named as ’backprop’. For the forward pass, we compute the sublinear layer as z = wx + b and then apply the Sigmoid activation function for the last layer and ReLu activation function for all other layers to compute the activations, given by a = Relu(z) or a = Sigmoid(z). For the backward pass, we compute the error term for the last layer by multiplying the cost gradient and the gradient for the Sigmoid activation function. For all other layers from the second last layer to the first layer, we compute the error term by multiplying the cost gradient and the gradient for the Relu activation function. Then, we update the weights and bias in the backward pass given by: dw<sup>(l)</sup> = e<sup>(l)</sup> z<sub>(l−1)</sub><sup>T</sup> and db<sup>(l)</sup> = e<sup>(l)</sup>. This completes the backprop.

Now, to train the model, we first create a network with 2 hidden layers given by Network([784, 30, 30, 10]) where the first parameter is the dimension of the input data and hence the first layer is the input layer, the second and third layers correspond to the hidden layers with 30 neurons each, and the fourth layer corresponds to the output layer with 10 nodes corresponding to the 10 class labels. Next, we call the sgd function which creates mini batches for the data and passes to the ’update mini batch’ function. The update mini batch function initiates backpropagation for each data point in the mini batch and updates the weights and biases of the network. Finally, the error and accuracy are calculated in the sgd function and are printed with respect to each epoch. The losses and accuracies of each epoch are also appended in an array and returned so that the learning curve of the model can be visualized. After trial and error, the best results for this model were obtained by keeping the learning rate = 1.0, the batch size = 10, and the number of epochs were restricted to 10 for faster training of the model. As a side note, the learning rate reduces to ’0.1’ since the weights and biases are updated by a factor of (learning rate/batch size) . A higher accuracy can be achieved by increasing the number of epochs and lowering the learning rate which could result in slow computations. The results for the model were achieved as given below where we obtain an accuracy of 95% on the train data and 94% on the test data.

(add image)

The loss curves for the train and test data were obtained as:

(add image)

(add image)

Here, as we see, the loss for the train and test data decreases monotonically over each epoch and thus increases the performance of the model.

Similarly, we can see that the accuracies of both the train and test data also increase monotonically, thus signifying an increase in the performance of the model.

(add image)

(add image)
