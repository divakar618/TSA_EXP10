# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


data = pd.read_csv('powerconsumption.csv')
data['date'] = pd.to_datetime(data['date'])


plt.plot(data['date'], data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.show()

def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['temp'])

plot_acf(data['temp'])
plt.show()
plot_pacf(data['temp'])
plt.show()

sarima_model = SARIMAX(data['temp'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['temp'][:train_size], data['temp'][train_size:]

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```


### OUTPUT:
![318165274-49964701-62a0-451a-9a12-61c6c960cd7c](https://github.com/user-attachments/assets/14647f66-12bc-431c-bf3c-1bf6b850bc5e)

![318165416-1738ccc5-7034-41f0-8f61-5d066d0471b5](https://github.com/user-attachments/assets/57cc2f7d-b682-443e-90ea-4927b84bea4e)

![318165474-a379bb63-dc9f-4882-9c21-906c805f806b](https://github.com/user-attachments/assets/1ca01a15-f8c2-4229-9546-f938b64cc271)

![318165479-0e558564-4b41-4bed-93ec-770953a9fd84](https://github.com/user-attachments/assets/53aa184b-04ce-4466-85fa-f7cd23e62778)


### RESULT:
Thus the program run successfully based on the SARIMA model.
