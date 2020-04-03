import numpy as np
import pickle


class FeedForward:
    """FeedForward"""

    networkLayers = np.array([])
    activation = ""
    totalNumNodes = 0
    net = np.array([])
    weights = np.array([])
    biasWeights = np.array([])
    values = np.array([])
    

    def __init__(self, networkLayers, activation, dtype=np.float64):
        """__init__.
       
        Parameters
        ----------
        networkLayers : [type]
            [description]
        activation : [type]
            [description]
        dtype : data type objects, optional
            The data-type objects (dtype) can be set to single (np.float32) or double (np.float64) precission, by default np.float64       
        """
        self.networkLayers = []
        startNode = 0
        endNode = 0
        for layer, numNodes in enumerate(networkLayers):
            if layer > 0:
                startNode += networkLayers[layer - 1]
            endNode += numNodes
            self.networkLayers.append(
                {
                    "num_nodes": numNodes,
                    "start_node": startNode,
                    "end_node": endNode - 1,
                }
            )

        self.totalNumNodes = np.sum(networkLayers, dtype=np.int)
        self.activation = activation
        self.dtype = dtype

    def initialise(self):
        """initialise bias-neurons and their weights.
        
        Bias-neurons- and their weights-matrices will be initialise by using np.zeros. The data-type (dtype) can be set 
        in the __init__.py, and can be choosen between single (np.float32) and double (np.float64) precission.
        """
        self.weights = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )
        self.biasWeights = np.zeros(
            (self.totalNumNodes, self.totalNumNodes), dtype=self.dtype
        )

        self.initialiseValuesNet()
        self.initialiseWeights()

    def initialiseValuesNet(self):
        """initialiseValuesNet values- and  net-array.
        """

        self.values = np.zeros(self.totalNumNodes, dtype=self.dtype)
        self.net = np.zeros(self.totalNumNodes, dtype=self.dtype)

    def initialiseWeights(self, low=-5, high=+5):
        """initialiseWeights for the weights and the bias-weights.
        
        Based on the zero-matrix, the weights- and bias-weights-matrix will be filled with random-int, which
        becomes float by np.divide. 
        
        Notes
        -----
        np.divide is important because it will also keep the dtype-format consistent.
        
        Parameters
        ----------
        low : int, optional
            lowest random-value, by default -5
        high : int, optional
            highes random-value, by default +5
        """

        self.weights = np.divide(
            np.random.randint(low=low, high=high, size=self.weights.shape),
            100.0,
            dtype=self.dtype,
        )
        self.biasWeights = np.divide(
            np.random.randint(low=low, high=high, size=self.biasWeights.shape),
            100.0,
            dtype=self.dtype,
        )

    def activate(self, inputs):
        """activate the forward propagation.
        
        For activate the forward propagation, the values(dendrites) will be elementwise combined with the 
        weights (synapes) plus the bias. This will be performed by numpy. Furthermore, the activation functions
        will be activated to the values.
        
        Notes:
        ------
        A more detail description is provided by Deep Learning: Ian Goodfellow et al. page 205
        
        Parameters
        ----------
        inputs : array
            inputs as float array to be processed
        """
        # Defining the h^0 = x
        _end = self.networkLayers[0]["num_nodes"]
        self.values[0:_end] = inputs[0:_end]

        # Connecting the layers (input, hidden, and target) via j-index
        for k, layer in enumerate(self.networkLayers[1:]):
            # Prevouis layer
            i_0, i_1 = self.getPreviousLayer(self.networkLayers, index=k)

            # Current layer
            j_0, j_1 = self.getCurrentLayer(layer)

            # Apply feedback transformation

            self.net[j_0:j_1] = np.add(
                self.biasWeights[k, j_0:j_1],
                np.dot(self.values[i_0:i_1], self.weights[i_0:i_1, j_0:j_1]),
            )

            # Apply activation function
            self.values[j_0:j_1] = self.activation.getActivation(self.net[j_0:j_1])

    def getOutputs(self):
        """getOutputs returns the predicted values
        
        Returns
        -------
        array
            Returns the predicted values as float array
        """
        startNode = self.networkLayers[len(self.networkLayers) - 1]["start_node"]
        endNode = self.networkLayers[len(self.networkLayers) - 1]["end_node"]
        return self.values[startNode : endNode + 1]
    
 

    def getNetworkLayers(self):
        """getNetworkLayers.
        
        Returns
        -------
         : dict 
            Dictonary of the network-layers including total-, start-, and end-number of nodes.
        """
        return self.networkLayers

    def getValue(self):
        """getValue.
        
        Returns
        -------
         : array
            All values as float-array
        """
        return self.values

    def getValueEntry(self, index):
        """getValueEntry.
        
        Returns the values for a explitic list of array-indices
        
        Parameters
        ----------
        index : list
            List of the array-indices to select
        
        Returns
        -------
        : array
            List of crop array-entries as array
        """
        return self.values[index]

    def getActivation(self):
        """getActivation [summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.activation

    def getNet(self):
        """getNet returns the net.
        
        Returns
        -------
        : array
            float-array of the net
        """
        return self.net

    def getNetEntry(self, index):
        """getNetEntry entry-value of the net.
        
        getNetEntry returns the current entry of the net based on the current index.
        Parameters
        ----------
        index : list
            List of the array-indices to select
        
        Returns
        -------
         : float 
            entry of the net based on the index
        """
        return self.net[index]

    def getWeight(self):
        """getWeight return the weight.
        
        Returns
        -------
        : array
            float-array of the weights
        """
        return self.weights

    def getWeightEntry(self, index):
        """getWeightEntry entry-value of the weight.
        
        [extended_summary]
        
        Parameters
        ----------
        index : list
            List of the array-indices to select
        
        Returns
        -------
         : float 
            entry of the net based on the index
        """
        return self.weights[index]

    def getBiasWeight(self):
        """
        getBiasWeights [summary]
        
        [extended_summary]
        
        Returns
        -------
        [type]
            [description]
        """
        return self.biasWeights

    def getBiasWeightEntire(self, index):
        """
        getBiasWeightEntire [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        index : list
            List of the array-indices to select
        
        Returns
        -------
        [type]
            [description]
        """
        return self.biasWeights[index]

    def setBiasWeights(self, biasWeights):
        """
        setBiasWeights [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        biasWeights : [type]
            [description]
        """
        self.biasWeights = biasWeights

    def updateWeight(self, i, j, weight):
        """updateWeight
        
        Parameters
        ----------
        i : list
            int-list of the column-entries
        j : list
            int-list of the row-entries
        weight : array
            updated weight-coefficients
        """
        # self.weights[i, j] += weight
        self.weights[i[0] : i[1], j[0] : j[1]] = np.add(
            self.weights[i[0] : i[1], j[0] : j[1]], weight, dtype=self.dtype
        )

    def updateBiasWeight(self, i, j, weight):
        """
        updateBiasWeight [summary]
        
        [extended_summary]
        
        Parameters
        ----------
        i : list
            int-list of the column-entries
        j : list
            int-list of the row-entries
        weight : array
             updated bias-weight-coefficients
        """
        # self.biasWeights[i, j] += weight
        if isinstance(i, int):
            self.biasWeights[i, j[0] : j[1]] = np.add(
                self.biasWeights[i, j[0] : j[1]], weight, dtype=self.dtype
            )
        else:
            self.biasWeights[i[0] : i[1], j[0] : j[1]] = np.add(
                self.biasWeights[i[0] : i[1], j[0] : j[1]], weight, dtype=self.dtype
            )

    def getTotalNumNodes(self):
        """getTotalNumNodes.
        
        Returns
        -------
         : int
            Total number of nodes as int
        """
        return self.totalNumNodes

    def getDtype(self):
        """getDtype."""
        return self.dtype

    def save(self, filename):
        """save the trained MLP-network.
        
        Saved the MLP-network as binary `pickle`-file.
        
        Parameters
        ----------
        filename : str
            filename of the to save pickle-file
        """
        with open(filename, "wb") as network_file:
            pickle.dump(self, network_file)

    @staticmethod
    def getPreviousLayer(layer, index, ext=+1):
        """getPreviousLayer start and end point of previous layer.
        
        Parameters
        ----------
        layer : dict
            Dictonary of the previous layer
        index : int
            index to select of the previous layer
        ext: int, optional
            Extend the layer-range, by default +1

        
        Returns
        -------
        i_0 : int
            startpoint of the previous layer
        
        i_1 : int
            endpoint of the previous layer
        """
        i_0, i_1 = layer[index]["start_node"], layer[index]["end_node"] + ext
        return i_0, i_1

    @staticmethod
    def getCurrentLayer(layer):
        """getCurrentLayer start and end point of current layer.
        
        Parameters
        ----------
        layer : dict
            Dictonary of the current layer
        
        Returns
        -------
        i_0 : int
            startpoint of the current layer
        
        i_1 : int
            endpoint of the current layer
        """
        i_0, i_1 = layer["start_node"], layer["end_node"] + 1
        return i_0, i_1

    @staticmethod
    def load(filename):
        """load a trained MLP-network.
        
        load a pre-trained MLP-network from binary `pickle`-file.
        
        Parameters
        ----------
        filename : str
            filename of the to load pickle-file
        
        Returns
        -------
         : array
            gives back the weight and bias coefficients of the MLP-network
        """
        with open(filename, "rb") as network_file:
            network = pickle.load(network_file)
            return network
