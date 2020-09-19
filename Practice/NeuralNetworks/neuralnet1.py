from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Same numbers generated every run:
        random.seed(1)

        # Creating Sinle neuron 
        # Range [-1;1]
        # Assigning random weights to the 1x3 matrix
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # We pass weighted neurons through Sigmoid function.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Sigmoid function's derivative, tells us if the weight is comfortable.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Trials and errors:
    # Adjusting the neuron connection's weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            
            output = self.think(training_set_inputs)

            # Calculating the error: difference between expected result and the result we have.
            error = training_set_outputs - output

            # Multiply the error by the input and Sigmoid function.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network, single neuron
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

#Main method:
if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights

    #Initializing some training set.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set:
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights

    print neural_network.think(array([1, 0, 0]))
 
