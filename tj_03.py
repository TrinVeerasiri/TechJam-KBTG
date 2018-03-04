###############################################################################
#ขั้นที่ 1: import และโหลดข้อมูล
###############################################################################

#import ข้อมูล (ข้อมูลนี้ใส่ log มาแล้ว)
import pandas as pd
import numpy as np

filename_train = 'p3_train.csv'
filename_test = 'p3_test.csv'

train = pd.read_csv(filename_train)
test = pd.read_csv(filename_test)

#เติมข้อมูลที่เป็น nan
train = train.fillna(train.mean())
test = test.fillna(train.mean())

#แยก train และ target
target = train['label']
train = train.drop(['account_no', 'label'], axis=1)
test = test.drop(['account_no'], axis=1)

#ใส่ log มามี -inf (ใน column ที่เป็น sd, เรา drop column นั้นทิ้งก่อน)
train = train.drop(['log_sd_amt', 'log_sd_amt.DR',\
                   'log_sd_amt_month.DR', 'log_sd_amt_day.DR',\
                   'log_sd_amt.CR', 'log_sd_amt_month.CR',\
                   'log_sd_amt_day.CR'], axis=1)
test = test.drop(['log_sd_amt', 'log_sd_amt.DR',\
                   'log_sd_amt_month.DR', 'log_sd_amt_day.DR',\
                   'log_sd_amt.CR', 'log_sd_amt_month.CR',\
                   'log_sd_amt_day.CR'], axis=1)

#test preprocess
#account_info = pd.read_csv('tj_03_account_info.csv')
#deposit_txn = pd.read_csv('tj_03_deposit_txn.csv')


###############################################################################
#DEEP
###############################################################################

from keras import optimizers
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras import optimizers

#แปลงเป็น array เพราะ keras รับ dataframe ไม่ได้
train = train.values
test = test.values

#Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)

input_dim = train.shape[1]
output_dim = 1

#โมเดลของตฤณ
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim))
model.add(Activation('sigmoid'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
#model.fit(train, target, batch_size = 10, epochs = 1000)
model.fit(train, target, batch_size=10, epochs = 300, validation_split=0.2)

#ทำนายผล
y_pred = model.predict(test)
#แปลงค่าคำตอบให้มีแค่ 0 หรือ 1 เท่านั้น
y_round = np.around(y_pred).astype(int)

#save เป็น .txt
np.savetxt('3.txt', y_round, fmt="%d")
