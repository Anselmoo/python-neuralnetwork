# Neural network
Artificial neural network for Python. Features online backpropagtion learning using gradient descent, momentum, the sigmoid and hyperbolic tangent activation function.

## About
The library allows you to build and train multi-layer neural networks. You first define the structure for the network. The number of input, output, layers and hidden nodes. The network is then constructed. Interconnection strengths are represented using an adjacency matrix and initialised to small random values.  Traning data is then presented to the network incrementally. The neural network uses an online backpropagation training algorithm that uses gradient descent to descend the error curve to adjust interconnection strengths. The aim of the training algorithm is to adjust the interconnection strengths in order to reduce the global error. The global error for the network is calculated using the mean sqaured error. 

You can provide a learning rate and momentum parameter.  The learning rate will affect the speed at which the neural network converges to an optimal solution. The momentum parameter will help gradient descent to avoid converging to a non optimal solution on the error curve called local minima.  The correct size for the momentum parameter will help to find the global minima but too large a value will prevent the neural network from ever converging to a solution.

Trained neural networks can be saved to file and loaded back for later activation.

## Installation
```bash
$  pip install neuralnetwork
```
## Examples
### Training XOR function on three layer neural network with two inputs and one output
```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

sigmoid = Sigmoid()

networkLayer = [2,2,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.3)

trainingSet = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)

    if(result):
        break

feedForward.activate([0,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([0,1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,1])
outputs = feedForward.getOutputs()
print(outputs[0])
```

### Training XOR function on three layer neural network using Hyperbolic Tangent
```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.HyperbolicTangent import HyperbolicTanget
from neuralnetwork.Backpropagation import Backpropagation

hyperbolicTangent = HyperbolicTangent()

networkLayer = [2,2,1]

feedForward = FeedForward(networkLayer, hyperbolicTangent)

backpropagation = Backpropagation(feedForward,0.7,0.3,0.001)

trainingSet = [
                    [-1,-1,-1],
                    [-1,1,1],
                    [1,-1,1],
                    [1,1,-1]
                ];

while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)

    if(result):
        break

feedForward.activate([-1,-1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([-1,1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,-1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,1])
outputs = feedForward.getOutputs()
print(outputs[0])
```

### Saving trained neural network to file
```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

sigmoid = Sigmoid()

networkLayer = [2,2,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.3)

trainingSet = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)

    if(result):
        break

feedForward.save('./network.txt')
```

### Load trained neural network from file
```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

feedForward = FeedForward.load('./network.txt')

feedForward.activate([0,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([0,1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,1])
outputs = feedForward.getOutputs()
print(outputs[0])
```

## Training Neural Network to Predict Diabetes
For this example we will train a neural network to predict whether a patient will develop diabetes within the next five years given various health measurements such as number of times pregnant, glucose plasma, blood pressure diastolic, skin fold thickness, serum insulin, body mass index, diabetes pedigree function, and age.

The dataset for this project is hosted by Kaggle. To download the necessary dataset for this example, please follow the instructions below.

1. Go to https://www.kaggle.com/uciml/pima-indians-diabetes-database

2. Click on the 'Download All' button

3. Kaggle will prompt you to sign in or to register. If you do not have a Kaggle account, you can register for one.

4. Upon signing in, the download will start automatically.

5. After the download is complete, unzip the zip file and move the file 'diabetes.csv' into your project folder.

For this example we first preprocess the data to ensure we have no missing or zero values.  We then scale or normalise the inputs before passing into the neural network.

We then split the data into a training and validation set which we use to test the accuracy of the trained neural network.

We then construct a neural network with 8 inputs, 32 nodes in the first hidden layer, 16 nodes in the second hidden layer, and finally one node in the output layer.

The neural network will output a 1 if the patient will develop diabetes and a 0 otherwise.

We then train the neural network using the training set.

Finally we test the trained neural network using the validation set and output and acurracy percetange.

```py
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import numpy as np
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
np.random.seed(16)

def preprocess(df):
    print('----------------------------------------------')
    print("Before preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col]==0].shape[0]
        print(col + ": " + str(missing_rows))
    print('----------------------------------------------')

    # Replace 0 values with the mean of the existing values
    df['Glucose'] = df['Glucose'].replace(0, np.nan)
    df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
    df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
    df['Insulin'] = df['Insulin'].replace(0, np.nan)
    df['BMI'] = df['BMI'].replace(0, np.nan)
    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
    df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

    print('----------------------------------------------')
    print("After preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col]==0].shape[0]
        print(col + ": " + str(missing_rows))
    print('----------------------------------------------')

    # Standardization
    df_scaled = preprocessing.scale(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    df_scaled['Outcome'] = df['Outcome']
    df = df_scaled
    

    return df



try:
    df = pd.read_csv('diabetes.csv')
except:
    print("""
      Dataset not found in your computer.
      Please follow the instructions in the link below to download the dataset:
      hhttps://github.com/stephenlmoshea/neural-network-to-predict-diabetes/blob/master/how_to_download_the_dataset.txt
      """)
    quit()

# Perform preprocessing and feature engineering
df = preprocess(df)

trainingSet = df.loc[:]

train, test = train_test_split(trainingSet, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [8,32,16,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.8, 0.05, 2000)

backpropagation.initialise()
result = backpropagation.train(train.values)

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test.values):
    feedForward.activate(row[:8])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[8]),round(outputs[0])))
    if(int(row[8]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))

```
