# F20BC-ANN
Coursework 1 

### Tasks:
Coding: 
Done:
1. Class Architecture -done
2. Forward Propogation (Logistic, Relu, Hyperbolic Tangent) -done
3. Backward Propogation (Binary Cross Entropy, 2 others) -done
6. Train-Test Split - done
7. K fold Cross-Validation - done
8. Implement one hot encoding when using 2 output nodes - done
11.Do normalization of the training data - done
12.Shuffle training data -done

To do:
- Batch GD - Rida
- Mini batch GD - Rida
- Accuracy for 2 output nodes - Rida
- Implementing dropout - Rida
- Train - test split - Rida
- Implement softmax act - Rida
- Cross validation testing - Bhavika
- Implementing learning rate schedule - Bhavika
- Graphical Application/ Console for user inputs - bhavika
- Accuracy for 1 output node (using a threshold value) - Bhavika


Report: 
1. Intro - Bhavika
2. Program Implementation Rationale (including reasons behind network architecture, literature review on activation and loss functions, etc) - Rida
3. Program Workflow(include class diagram?) - Rida
4. Results (basically just plotting visualizations, as mentioned in the specification and a simple reporting of the results - refer to visualisations below) - Bhavika
5. Discussions (interpreting the results above and giving reasons for different behaviors) - Bhavika
6. Conculsions

Visualisation: 
1. Evaluate how different activation functions change accuracy 
2. Evaluate how different loss functions change accuracy
3. Evaluate how different training schedules change accuracy
4. Evaluate how different number of nodes change accuracy (both output and hidden layer nodes)
5. Add more...

Note: Network now configurable for activations and no. of hidden layers/nodes. Currently using 3 hidden layers. 
1. input layer -> tanh, 30nodes
2. hidden 1 -> relu, 7 nodes
3. hidden 2 -> sigmoid, 5 nodes
4. hidden 3 -> sigmoid, 3 nodes
5. output layer -> NA, 2 nodes

## Dump your sources here:
1. https://hackernoon.com/deep-learning-feedforward-neural-networks-explained-c34ae3f084f1?ref=hackernoon.com
2. https://hackernoon.com/building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b
3. https://www.kdnuggets.com/2019/11/build-artificial-neural-network-scratch-part-1.html
4. https://towardsdatascience.com/feed-forward-neural-networks-how-to-successfully-build-them-in-python-74503409d99a
5. https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76
6. https://github.com/omaraflak/Medium-Python-Neural-Network/blob/master/network.py
7. https://github.com/jaymody/backpropagation/blob/master/nn.ipynb
8. Previous F20BC Repository: https://github.com/syedkhajahussainsa/F21BC--Biologically-Inspired-Computation-cw
9. https://www.bogotobogo.com/python/python_numpy_batch_gradient_descent_algorithm.php#:~:text=Gradient%20descent%20is%20an%20optimization,an%20assign%20or%20an%20update.
10. for report: https://www.d.umn.edu/~rmaclin/cs1511/fall1999/lab-report.html
11. report example: https://github.com/JZ76/GD-and-PSO/blob/main/F21BC%20Stage%201%20Report%20-%20Gabini%20%26%20Zhang.pdf
## For our ANN: 
### Activation Functions: 
- Logistic
- ReLU
- Hyperbolic Tanget
- Sigmoid
### Overfitting (extra / not required)
- Train/Test split and cross validation
### HyperParameters
- Input Nodes:
- Output Nodes: 1 (binary)
- Hidden Layers: 2 probably
- Activation function in each layer
- Learning rate
- Loss fuction: Cross Entropy
- Gradient Descent types: all 3?

column names: 
![image](https://user-images.githubusercontent.com/97593074/195399089-962ca97b-ef0a-4333-a1e0-1d0da0e51822.png)

