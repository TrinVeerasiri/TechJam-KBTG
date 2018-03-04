###############################################################################
#ขั้นที่ 1: import และโหลดข้อมูล
###############################################################################

#import ข้อมูล
import pandas as pd
import numpy as np

filename_train = 'p2_train.csv'
filename_test = 'p2_test.csv'

train = pd.read_csv(filename_train)
test = pd.read_csv(filename_test)

#เติมข้อมูลที่เป็น nan
train = train.fillna(train.mean())
test = test.fillna(train.mean())

#แยก train และ target
target = train['label']
train = train.drop(['account_no', 'label'], axis=1)
test = test.drop(['account_no'], axis=1)



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

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_card_flag_train = LabelEncoder()
train[:,41] = labelencoder_card_flag_train.fit_transform(train[:, 41])
labelencoder_card_flag_test = LabelEncoder()
test[:,41] = labelencoder_card_flag_test.fit_transform(test[:, 41])

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

adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
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
np.savetxt('2.txt', y_round, fmt="%d")
