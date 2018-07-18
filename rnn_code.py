# Build Recurrent Neural Network
'''
Use LSTM to build RNN model that will try to show upward and downward trend of google stock price.
We are going to train LSTM model on 5 years google stock price from 2012 to 2016 and based on this correlation we will try to predict stock price of Jan 2017.
Note here we not predicting exact stock price instead we are trying to predict upward or downward trend for Jan 2017 month. 
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 

# Importing training set
'''
We need to import training set and make that numpy array because only numpy arrays can be input of neural networks in keras.
'''
dataset_train = pd.read_csv("C:\\Users\\Chandan.S\\Desktop\\DeepLearning\\RNN\\Google_Stock_Price_Train.csv")
'''
training_set = dataset_train.iloc[:,1] Here we cannot input only index 1, because we want to create numpy array not a simple vector.
This can be done by taking range of index 1:2. And this will have numpy array of one column.
dataset_train.iloc[:,1:2] - This will create a data frame of column "open" and to make it numpy array use ".values".
'''
training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling
'''
Two best techniques for applying feature scaling are :
1) Standardisation and
2) Normalisation. Here we are using normalistion, if RNN is having sigmoid activation function in output layer of RNN,
then it is always recommanded to used normalisation. That can be done using MinMaxScaler class.
'''
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure 
'''
120 timestamp and 1 output : 120 timestamp means that at each time "t" RNN is going to take 120 stock price before time "t" and try to predict stock price at time "t".
RNN will take 120 past information and try to learn some correlation and based on that correlation it will try to predict next output.
Here number 120 came from experiment, we can other numbers also (20,30,40,50,60...120).
x_train : 120 previous stock price
y_train : next stock price i.e stock price at time t+1
'''
x_train = []
y_train = []
for i in range(120, 1258):
    x_train.append(training_set_scaled[i-120:i, 0])
    y_train.append(training_set_scaled[i, 0])

# convert x_train and y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)


# Reshaping data : Adding more dimension.
'''
Dimension we are adding is unit that is number of prdictors we can use to predict stock price at time "t+1".
This new dimension could be new indicator from data set that can help in prediction.  

https://keras.io/layers/recurrent/
Input shape
3D tensor with shape (batch_size, timesteps, input_dim)
batch_size = x_train.shape[0]
timesteps = x_train.shape[1]
input_dim = 1
'''

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building the RNN as sequence of layers. Sequential() to initialize RNN.
regressor = Sequential()

# Adding first LSTM layer to RNN
'''
1) units = no of memory units you want to have in LSTM or number of LSTM cells
2) return_sequences will be set to "True" because we are building stacked RNN with multiple layers.
If you want to add new LSTM layer after current layer then return_sequences = True and
if it is last layer then return_sequences will be set to False
3) input_shape = Shape of x_train, but here we need not to give 3D shape, only shape corresponding to 
timestamps(2nd) and indicators(3rd) are needed. Shape corresponding to observation(1st) will automatically taken into account.
x_train.shape[0]: Shape corresponding to observation
x_train.shape[1]: Shape corresponding to timestamp
1: Shape corresponding to predictors

'''
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))

#Add Dropout to avoid over fitting
regressor.add(Dropout(rate=0.2)) 

# Add 2nd LSTM layer to RNN
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(rate=0.2)) 

# Add 3rd LSTM layer to RNN
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Add 4th LSTM layer to RNN
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Add 5th LSTM layer to RNN
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Add 6th LSTM layer to RNN
# We are not adding 7th LSTM layer so setting return_sequences = False, which is the default value hence removed.
regressor.add(LSTM(units=100))
regressor.add(Dropout(rate=0.2))

# Adding output layer
# Output layer is fully connected layer hence use Dense
regressor.add(Dense(units=1))

# Compile RNN
regressor.compile(optimizer= 'adam',loss='mean_squared_error')

# Till now we have built and compiled RNN, next step is to fit this RNN to training set (x_train, y_train)
# Fit RNN to training set
regressor.fit(x_train, y_train, epochs=100, batch_size=32)


# Prediction and visualization of result
# Real Stock Price
dataset_test = pd.read_csv("C:\\Users\\Chandan.S\\Desktop\\DeepLearning\\RNN\\Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values

# Prediction for Jan 2017 Google stock price
# 3 important points mentioned below:
'''
1) We trained our model to predict stock price at time "t+1" based on 120 previous stock price. 
So in order to predict stock price on each finanical day of Jan 2017 we will need stock price of 120 previous
financial days.
2) In order to get 120 previous day stock price we will need both training and test set. Because we will have some days from Dec 2016 and some days from Jan 2017
So to get that we need to concate traning and test set. 
3) We cant concate training_set and real_stock_price because we need to apply scaling and after scaling acutal test set will change. We should alway avoid changing actual test set.
So in that case we will use dataset_train and dataset_test for concatenation.
** Here our RRN model is trained on scaled data hence we need to input data after concatenation.
'''
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

# first day of Jan 2017(3rd Jan 2017) = len(dataset_total) - len(dataset_test)
# Want to get stock price of 3rd Jan 2017 : Lower bound = len(dataset_total) - len(dataset_test) - 60
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range(120, 140):
    x_test.append(inputs[i-120:i, 0])

# convert x_train and y_train to numpy array
x_test = np.array(x_test)

# Reshape test data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicated_stock_price = regressor.predict(x_test)
predicated_stock_price = sc.inverse_transform(predicated_stock_price)

# Visualization 
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicated_stock_price, color = 'green', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Predication')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
