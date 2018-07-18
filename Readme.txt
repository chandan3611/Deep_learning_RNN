
RNN:
----------------------------------------------------------------------------

Predict tend for Google stock price.
Here we will use LSTM to build RNN model that will try to show upward and downward trend of google stock price

For find trend LSTM is very power full tool, it behave better than ARIMA model.

We are going to train LSTM model on 5 years google stock price from 2012 to 2016 and based on this correlation we will try to predict stock price of Jan 2017.
Note here we not predicting exact stock price instead we are trying to predict upward or downward trend for Jan 2017 month.


Structure: 
120 timestamp and 1 output : 120 timestamp means that at each time "t" RNN is going to take 120 stock price before time "t" and try to predict stock price at time "t".
RNN will take 120 past information and try to learn some correlation and based on that correlation it will try to predict next output.
Here number 120 came from experiment, we can other numbers also (20,30,40,50,60...120).
x_train : 120 previous stock price
y_train : next stock price i.e stock price at time t+1

LSTM Layer: Added 6 LSTM layers each of 100 units

----------------------------------------------------------------------------------
DataSet : Google_Stock_Price_Train.csv and Google_Stock_Price_Test.csv



 