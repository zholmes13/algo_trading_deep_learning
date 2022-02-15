from tensorflow.keras.utils import to_categorical
import config
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta


from datetime import date

train_start = date(2017, 1, 1)
train_end = date(2019, 10, 31)
train_delta = train_end - train_start
print(f'Number of days of Training Data {train_delta.days}')

val_day_num = 400
print(f'Number of days of Validation Data {val_day_num}')

test_start = train_end + timedelta(val_day_num)
test_end = date.today()
test_delta = (test_end - test_start)
print(f'Number of days of Holdout Test Data {test_delta.days}')

ticker = "HD" # Ticker Symbol to Test
interval = "5Min" # Interval of bars
train_day_int = train_delta.days # Size of training set (Jan 2010 - Oct 2017)
val_day_int = val_day_num # Size of validation set
test_day_int = test_delta.days # Size of test set
offset_day_int = 0 # Number of days to off set the training data
train, val, test, full, offset, complete, traintest_day, testval_day = prepost_train_test_validate_offset_data(api, ticker,
                                                                                                               interval,
                                                                                                               train_days=train_day_int,
                                                                                                               test_days=test_day_int,
                                                                                                               validate_days=val_day_int,
                                                                                                               offset_days = offset_day_int)


train = timeFilterAndBackfill(train)
val = timeFilterAndBackfill(val)
test = timeFilterAndBackfill(test)
print(train.shape)

train = train[train.index.dayofweek <= 4].copy()
val = val[val.index.dayofweek <= 4].copy()
test = test[test.index.dayofweek <= 4].copy()
print(train.shape)

train["Open"] = np.where((train["Volume"] == 0), train["Close"], train["Open"])
train["High"] = np.where((train["Volume"] == 0), train["Close"], train["High"])
train["Low"] = np.where((train["Volume"] == 0), train["Close"], train["Low"])

val["Open"] = np.where((val["Volume"] == 0), val["Close"], val["Open"])
val["High"] = np.where((val["Volume"] == 0), val["Close"], val["High"])
val["Low"] = np.where((val["Volume"] == 0), val["Close"], val["Low"])

test["Open"] = np.where((test["Volume"] == 0), test["Close"], test["Open"])
test["High"] = np.where((test["Volume"] == 0), test["Close"], test["High"])
test["Low"] = np.where((test["Volume"] == 0), test["Close"], test["Low"])

train_tonp = train[["Open", "High", "Low", "Close", "Volume"]]
val_tonp = val[["Open", "High", "Low", "Close", "Volume"]]
test_tonp = test[["Open", "High", "Low", "Close", "Volume"]]

train_array = train_tonp.to_numpy()
val_array = val_tonp.to_numpy()
test_array = test_tonp.to_numpy()

print(train_array.shape)

X_train = blockshaped(train_array, 24, 5)
X_val = blockshaped(val_array, 24, 5)
X_test = blockshaped(test_array, 24, 5)

volity_val = 10
y_train = buildTargets(X_train, volity_int = volity_val)
y_val = buildTargets(X_val, volity_int = volity_val)
y_test = buildTargets(X_test, volity_int = volity_val)

# remove all of the last two hour increments from the data, we will not predict
# movements for the next day
X_train = np.delete(X_train, np.arange(3, X_train.shape[0], 4),axis=0)
X_val = np.delete(X_val, np.arange(3, X_val.shape[0], 4),axis=0)
X_test = np.delete(X_test, np.arange(3, X_test.shape[0], 4),axis=0)

y_train = np.delete(y_train, np.arange(3, y_train.shape[0], 4),axis=0)
y_val = np.delete(y_val, np.arange(3, y_val.shape[0], 4),axis=0)
y_test = np.delete(y_test, np.arange(3, y_test.shape[0], 4),axis=0)

print(y_train.shape)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,7))
# Train
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Class Distribution in Train Set')
# Validation
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Class Distribution in Val Set')
# Test
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[2]).set_title('Class Distribution in Test Set')

volatility = buildTargets_VolOnly()

fig = plt.figure(figsize=(15, 7))
ax1 = fig.add_subplot(1, 1, 1)
volatility.plot(ax=ax1, color = "red")
ax1.set_xlabel('Date')
ax1.set_ylabel('Volatility', color = "red")
ax1.set_title(f'Annualized volatility for {ticker}')
ax2 = ax1.twinx()
full.Close.plot(ax=ax2, color = "blue")
ax2.set_ylabel('Close', color = "blue")
ax2.axvline(x=full.index[train.shape[0]])
ax2.axvline(x=full.index[val.shape[0]+train.shape[0]])
plt.show()

X_train_new = np.zeros([X_train.shape[0],X_train.shape[2],X_train.shape[1]])
for i, x in enumerate(X_train):
    X_train_new[i] = x.T

X_train = X_train_new

X_val_new = np.zeros([X_val.shape[0],X_val.shape[2],X_val.shape[1]])
for i, x in enumerate(X_val):
    X_val_new[i] = x.T

X_val = X_val_new

X_test_new = np.zeros([X_test.shape[0],X_test.shape[2],X_test.shape[1]])
for i, x in enumerate(X_test):
    X_test_new[i] = x.T

X_test = X_test_new

print(f'X Train Length {X_train.shape}, y Train Label Length {y_train.shape}')
print(f'X Val Length {X_val.shape}, y Val Label Length {y_val.shape}')
print(f'X Test Length {X_test.shape}, y Test Label Length {y_test.shape}')


y_train = to_categorical(y_train, 3)
y_val = to_categorical(y_val, 3)
y_test = to_categorical(y_test, 3)

y_train.shape

print(f'X Train Length {X_train.shape}, y Train Label Length {y_train.shape}')
print(f'X Val Length {X_val.shape}, y Val Label Length {y_val.shape}')
print(f'X Test Length {X_test.shape}, y Test Label Length {y_test.shape}')
print("")
print('Training data window: ', len(X_train))
print('Val data windows: ', len(X_val))

model = build_model(X_train)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy']) #, loss_weights=[1, 1, 100])

r = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, class_weight={0:2, 1:3, 2:1})

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

y_pred = model.predict(X_test)

# Calculate the accuracy
test_preds = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
test_acc = np.sum(test_preds == y_true)/y_true.shape[0]

# Recall for each class
recall_vals = []
for i in range(3):
    class_idx = np.argwhere(y_true==i)
    total = len(class_idx)
    correct = np.sum(test_preds[class_idx]==i)
    recall = correct / total
    recall_vals.append(recall)

classes = [0,1,2]
# Calculate the test set accuracy and recall for each class
print('Test set accuracy is {:.3f}'.format(test_acc))
for i in range(3):
    print('For class {}, recall is {:.3f}'.format(classes[i],recall_vals[i]))

print("Weighted F score is {:.3f}".format(calculate_weighted_f_score(y_true, y_pred)))
