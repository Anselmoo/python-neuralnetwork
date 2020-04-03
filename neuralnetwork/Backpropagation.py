import logging
import numpy as np


class Backpropagation:
    """Backpropagation
    
    The Backpropagation class calculates the minimum value of the error function in relation to the training-set and the activation function.
    The technique for achieving this goal is called the delta rule or gradient descent. 
    
    """

    nodeDeltas = np.array([])
    gradients = np.array([])
    biasGradients = np.array([])
    learningRate = np.array([])
    eta = np.array([])
    weightUpdates = np.array([])
    biasWeightUpdates = np.array([])
    minimumError = ""
    maxNumEpochs = ""
    numEpochs = ""
    network = np.array([])
    delta = np.float64
    networkLayers = []
    error = 0.
    def __init__(
        self, network, learningRate, eta, minimumError=0.005, maxNumEpochs=2000
    ):
        """
        __init__ [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        network : class
            class of FeedForward-Routine
        learningRate : float
            Learning rate of the MLP
        eta : float
            Error correction factor
        minimumError : float, optional
            Minimal error to stop the training, by default 0.005
        maxNumEpochs : int, optional
            Maxinum numbers of epochs before stopping the training, by default 2000
        """
        self.network = network
        self.learningRate = learningRate
        self.eta = eta
        self.minimumError = minimumError
        self.maxNumEpochs = maxNumEpochs
        self.initialise()

    def initialise(self):
        """initialise MLP.
        
        The intiale procedure includes:
            1. network
            2. node deltas
            3. gradients of values
            4. gradients of bias
            5. Update matrices for:
                a. weight of values
                b. weight of bias
                c. gradients of values
                d. gradients of bias
             
        """
        self.network.initialise()
        self.nodeDeltas = np.array([])
        self.gradients = np.array([])
        self.biasGradients = np.array([])
        self.totalNumNodes = self.network.getTotalNumNodes()
        self.dtype = self.network.getDtype()
        self.networkLayers = self.network.getNetworkLayers()
        # initiale the weight, bias, and gradients matrices
        self.weightUpdates = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.biasWeightUpdates = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.gradients = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.biasGradients = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.initialiseValues()

    def initialiseValues(self):
        """
        initialiseValues inital the values array
        """
        self.nodeDeltas = np.zeros(self.totalNumNodes, dtype=self.dtype)

    def train(self, trainingSets, rprint=False):  
        """train the mlp-network.
        
        Training of the mlp-network for a given `trainingSets` for maximum number of epchos. 
        
        Parameters
        ----------
        trainingSets : array
            The training set is provided as float-array where X- and y-values are keeped together.
        rprint : bool, optional
            print the current progress with global error, by default False
        Returns
        -------
         : bool
            Return a bool for indicating successful (True) or failed (False) learning.
        """
     
        self.numEpochs = 1
        if rprint:
            logging.basicConfig(level=logging.INFO)
        # Have to change to a for-if slope
        while True:
            if self.numEpochs > self.maxNumEpochs:
                return False
            sumNetworkError = 0
            for i in range(len(trainingSets)):
                # Switching to FeedForworad.py
                self.network.activate(trainingSets[i])
                outputs = self.network.getOutputs()
                # Come back to Backpropagation.py
                self.calculateNodeDeltas(trainingSets[i])
                self.calculateGradients()
                self.calculateWeightUpdates()
                self.applyWeightChanges()
                sumNetworkError += self.calculateNetworkError(trainingSets[i])
            globalError = sumNetworkError / len(trainingSets)
            logging.info("--------------------------------")
            logging.info("Num Epochs: {}".format(self.numEpochs))
            logging.info("Global Error: {}".format(globalError))
            self.error = globalError
            self.numEpochs = self.numEpochs + 1
            if globalError < self.minimumError:
                break
        return True

    def calculateNodeDeltas(self, trainingSet):
        """calculateNodeDeltas, error of each node.
        
        
        Parameters
        ----------
        trainingSets : array
            The training set is provided as float-array where X- and y-values are keeped together.
        """
        idealOutputs = trainingSet[
            -1 * self.networkLayers[len(self.networkLayers) - 1]["num_nodes"]
        ]
        # Initial phase

        actl_node = [
            self.networkLayers[len(self.networkLayers) - 1]["start_node"],
            self.networkLayers[len(self.networkLayers) - 1]["end_node"] + 1,
        ]
        activation = self.network.getActivation()
        error = self.network.values[actl_node[0] : actl_node[1]] - idealOutputs

        self.nodeDeltas[actl_node[0] : actl_node[1]] = np.multiply(
            -error,
            activation.getDerivative(self.network.net[actl_node[0] : actl_node[1]]),
            dtype=self.dtype,
        )

        for k in range(len(self.networkLayers) - 2, 0, -1):

            actl_node = [
                self.networkLayers[k]["start_node"],
                self.networkLayers[k]["end_node"] + 1,
            ]
            connectNode = len(self.network.getWeight())
            # Calculating the node deltas
            self.nodeDeltas[actl_node[0] : actl_node[1]] = np.multiply(
                np.dot(
                    self.network.weights[actl_node[0] : actl_node[1]],
                    self.nodeDeltas[:connectNode],
                ),
                activation.getDerivative(self.network.net[actl_node[0] : actl_node[1]]),
                dtype=self.dtype,
            )

    def calculateGradients(self):
        """calculateGradients, gradient of each value and bias.
        """

        for num, layer in enumerate(self.networkLayers[:-1]):
            prev_index = [layer["start_node"], layer["end_node"] + 1]
            # similiar to i
            actl_index = [
                self.networkLayers[num + 1]["start_node"],
                self.networkLayers[num + 1]["end_node"] + 1,
            ]  # similiar to j
            # Value-Gradient
            self.gradients[
                prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
            ] = np.outer(
                self.network.values[prev_index[0] : prev_index[1]],
                self.nodeDeltas[actl_index[0] : actl_index[1]],
                # dtype=self.dtype,
            )
            # Bias-Gradient
            self.biasGradients[num, actl_index[0] : actl_index[1]] = self.nodeDeltas[
                actl_index[0] : actl_index[1]
            ]

    def calculateWeightUpdates(self):
        """calculateWeightUpdates of the 'new' weights and bias-weights.
        """
        for num, layer in enumerate(self.networkLayers[:-1]):
            prev_index = [layer["start_node"], layer["end_node"] + 1]
            # similiar to i
            actl_index = [
                self.networkLayers[num + 1]["start_node"],
                self.networkLayers[num + 1]["end_node"] + 1,
            ]  # similiar to j
            # Updating the weights
            self.weightUpdates[
                prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
            ] = np.add(
                np.multiply(
                    self.learningRate,
                    self.gradients[
                        prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
                    ],
                    dtype=self.dtype,
                ),
                np.multiply(
                    self.eta,
                    self.weightUpdates[
                        prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
                    ],
                    dtype=self.dtype,
                ),
                dtype=self.dtype,
            )
            # Updating the bias-weights
            self.biasWeightUpdates[num, actl_index[0] : actl_index[1]] = np.add(
                np.multiply(
                    self.learningRate,
                    self.biasGradients[num, actl_index[0] : actl_index[1]],
                    dtype=self.dtype,
                ),
                np.multiply(
                    self.eta,
                    self.biasWeightUpdates[num, actl_index[0] : actl_index[1]],
                    dtype=self.dtype,
                ),
                dtype=self.dtype,
            )

    def applyWeightChanges(self):
        """applyWeightChanges of the gradient correction to the layers.
        """
        for num, layer in enumerate(self.networkLayers[:-1]):
            prev_index = [layer["start_node"], layer["end_node"] + 1]
            # similiar to i
            actl_index = [
                self.networkLayers[num + 1]["start_node"],
                self.networkLayers[num + 1]["end_node"] + 1,
            ]  # similiar to j
            self.network.updateWeight(
                prev_index,
                actl_index,
                self.weightUpdates[
                    prev_index[0] : prev_index[1], actl_index[0] : actl_index[1]
                ],
            )
            self.network.updateBiasWeight(
                num,
                actl_index,
                self.biasWeightUpdates[num, actl_index[0] : actl_index[1]],
            )

    def calculateNetworkError(self, trainingSet):
        """calculateNetworkError based on the the mean squared error.
        
        
        calculateNetworkError is using the mean squared error (MSE) for measuring the average of the squares of the errors. 
        In this context, the average squared difference between the predicted values and the real values (training set).
        
        Parameters
        ----------
        trainingSet : array
            The training-set with X,y for validation of the optimization-cycle
        
        Returns
        -------
        globalError : float
            Global Error as a non-negative floating point value (the best value is 0.0); defined as MSE
        """
        idealOutputs = trainingSet[
            -1 * self.networkLayers[len(self.networkLayers) - 1]["num_nodes"]
        ]
        startNode = self.networkLayers[len(self.networkLayers) - 1]["start_node"]
        endNode = self.networkLayers[len(self.networkLayers) - 1]["end_node"]
        numNodes = self.networkLayers[len(self.networkLayers) - 1]["num_nodes"]

        globalError = np.mean(
            np.square(
                np.subtract(
                    idealOutputs,
                    self.network.values[startNode : endNode + 1],
                    dtype=self.dtype,
                ),
                dtype=self.dtype,
            ),
            dtype=self.dtype,
        )

        return globalError
    
    def getGlobalError(self):
        """
        getGlobalError [summary]
        
        Returns
        -------
        error : float
            MSE-based global error
        """
        return self.error
