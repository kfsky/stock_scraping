"""
data cleaning the stock data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

"""
select the stock data
scraping_data.py scrapes the multiple stock data.
"""
df = pd.read_csv("../scrape_data/data/3189-ＡＮＡＰ.csv")


# rename columns
df.rename(columns={'日付': 'Date', '始値': 'Open', '高値': 'High', '安値': 'Low',
                   '終値': 'Close', '出来高': 'Value', '終値調整': 'Adj Close'}, inplace=True)

"""
create the target variable
Predict whether the closing price will rise or fall compared to the previous day.
"""
df["Up"] = df["Close"].diff()
df["Up"].fillna(0, inplace=True)
df.loc[df["Up"] < 0, "Up"] = 0
df.loc[df["Up"] > 0, "Up"] = 1

# set index and sort
df["Date"] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df.sort_values(by='Date', ascending=True, inplace=True)
df.drop(columns="Unnamed: 0", inplace=True)

# drop Ajd Close
df.drop(columns=["Adj Close"], inplace=True)

# create body
df["Body"] = df["Open"] - df["Close"]
df_rate = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
df_rate = df_rate.fillna(0)

X_data = df.drop(columns=["Up"], inplace=False)
y_data = df["Up"]

# split train, val, test
X_trainval, X_test, y_trainval, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, shuffle=False)


def get_t(X, num_date):
    # 入力データをNumPy配列に変換
    X = np.array(X)
    X_t_list = []
    for i in range(len(X) - num_date + 1):
        X_t = X[i:i+num_date, :]
        X_t_list.append(X_t)
    # Numpy配列のreturn
    return np.array(X_t_list)


def get_standardized_t(X, num_date):
    X = np.array(X)
    X_t_list = []
    for i in range(len(X) - num_date + 1):
        X_t = X[i:i+num_date]
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X_t)
        X_t_list.append(X_standardized)
    return np.array(X_t_list)


# period
num_date = 5
X_t_list = get_t(X=X_train, num_date=num_date)
X_array_data = get_standardized_t(X=X_train, num_date=num_date)

X_train_t = get_standardized_t(X=X_train, num_date=num_date)
X_val_t = get_standardized_t(X=X_val, num_date=num_date)
X_test_t = get_standardized_t(X=X_test, num_date=num_date)

y_train_t = y_train[num_date-1:]
y_val_t = y_val[num_date-1:]
y_test_t = y_test[num_date-1:]

# Network size
num_l1 = 100
num_l2 = 20
num_output = 1

model = Sequential()

model.add(LSTM(units=num_l1,
                activation='tanh',
                batch_input_shape=(None, X_train_t.shape[1], X_train_t.shape[2])))

model.add(Dense(num_l2, activation='relu'))

model.add(Dense(num_output, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


result = model.fit(x=X_train_t, y=y_train_t, epochs=80, batch_size=24, validation_data=(X_val_t, y_val_t))


# 折れ線グラフによる学習データの正解率の描画
plt.plot(result.history["acc"])

# 折れ線グラフによる検証データの正解率の描画
plt.plot(result.history["val_acc"])

# 凡例の指定
plt.legend(['Train', 'Val'])

# グラフの軸タイトルの指定
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# 描画の実行
plt.show()
