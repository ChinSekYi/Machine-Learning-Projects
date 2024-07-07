# Project Description
In this project, we will train a convolutional neural network for ants and bees image classification using transfer learning.

## Dataset
- 120 training images each for ants and bees
- 75 validation images for each class
- This dataset is a very small subset of imagenet

## Important terms
Convolutional layers 
-  turn input signals of several channels into feature maps/activation maps 

Max Pooling 
- to down sample the dimensions of our image to allow for assumptions to be made about the features in certain regoins of the image 
- nn.MaxPool2d(2,2) -> turning our image into 2x2 dimensions while retaining important features  

Fully connected layers 
- every neuron in previous layers connects to all neurons in the next. Used linear transformation g(Wx + b). g = ReLU 

ReLU
- Activation function necessary in a CNN to introduce non-linearity to our network.  

CrossEntropyLoss  
- used when training classification problems 
- it combines log softmax and negative log-likelihood 
1. softmax - scales numbers into probablities for each outcome. (probabilities sum to 1) 
2. log softmax - applying log handles numerical unstability. Also allows for improved numerical performance and gradient optimization 
3. negative log likelihood - is a loss function that calculates the loss based on the range of its function. 


## Pytorch code
Pytorch's `optim.SGD` stochastic gradient desccent:
Arguments:
- `net.parameters()` -> gets the learnable parameters of the CNN   
- `lr` -> learning rate of the gradient descent (how big of a step to take)   
- `momentum` -> helps accelerate gradient vectors in the right directions, which leads to faster converging.  

 
Connection between `loss.backward()` and `optimiser.step()` 
- Recall that when initializing optimizer you explicitly tell it what parameters (tensors) of the model it should be updating.  
- The gradients are "stored" by the tensors themselves (they have a grad and a requires_grad attributes) once you call backward() on the loss.  
- After computing the gradients for all tensors in the model, calling optimizer.step() makes the optimizer iterate over all parameters (tensors) it is supposed to update and use their internally stored grad to update their values. 


Why zero the gradients `optimizer.zero_grad()`?   
- Once you've completed a step, you don't really need to keep track of your previous suggestion (i.e. gradients) of where to step. By zeroing the gradients, you are throwing away this information. Some optimizers already keep track of this information automatically and internally.   
- With the next batch of inputs, you begin from a clean slate to suggest where to step next. This suggestion is pure and not influenced by the past. You then feed this "pure" information to the optimizer, which then decides exactly where to step.   
- Of course, you can decide to hold onto previous gradients, but that information is somewhat outdated since you're in an entirely new spot on the loss surface. Who is to say that the best direction to go next is still the same as the previous? It might be completely different! That's why most popular optimization algorithms throw most of that outdated information away (by zeroing the gradients).   

## Reference:
- <a href="https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html">Transfer learning using Pytorch</a>