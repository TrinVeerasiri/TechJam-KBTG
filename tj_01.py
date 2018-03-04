###############################################################################
#ขั้นที่ 1: import และโหลดข้อมูล
###############################################################################

#import ข้อมูล
import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.stats import skew
import matplotlib.pyplot as plt

p1train = 'p1_train.csv'
p1test = 'p1_test.csv'

p1train = pd.read_csv(p1train)
p1test = pd.read_csv(p1test)

card = 'tj_01_creditcard_card.csv'
customer = 'tj_01_creditcard_customer.csv'
transaction = 'tj_01_creditcard_transaction.csv'
test = 'tj_01_test.csv'
train = 'tj_01_training.csv'

card = pd.read_csv(card)
customer = pd.read_csv(customer)
transaction = pd.read_csv(transaction)
test = pd.read_csv(test, names = ['card_no'])
train = pd.read_csv(train, names = ['card_no', 'npl_flag'])

train = p1train
test = p1test


###############################################################################
#THE ARMOURY
###############################################################################
#Merge ข้อมูล
x = pd.merge(left=train, right=card, left_on='card_no', right_on='card_no')
x = pd.merge(left=x, right=customer, left_on='cst_id', right_on='cst_id')
train = pd.merge(left=x, right=transaction, left_on='card_no', right_on='card_no')

#สร้าง Histogram plt.hist()
#นับจำนวน pd.value_counts()

#นับ unique element (ตัดตัวที่ซ้ำออก) ไม่รวม NA values
transaction['card_no'].nunique()
transaction['card_no'].value_counts()

#ทดสอบ data aggregation
#สร้าง column ใน dataframe ชื่อ 'tran_count' 
df['tran_count'] = transaction['card_no'].value_counts()
df=transaction.groupby('card_no', as_index=False).agg({'card_no':'count', "txn_amount": "mean"})

#ทดลองดึงข้อมูลที่เป็น date
year = card['pos_dt'].str[0:4]
month = card['pos_dt'].str[5:7]
date = card['pos_dt'].str[8:10]

#ทำให้เป็น int
#int 64-bit
year=year.astype(np.int64)
#int 32-bit
month=month.astype(int)

#นับจำนวนวัน
card['pos_dt'].dtype
card['open_dt'].dtype

card['diff_date'] = pd.to_datetime(card['pos_dt']) - pd.to_datetime(card['open_dt'])
card['diff_date'] = card['diff_date'].dt.days

###############################################################################


###############################################################################
#ขั้นที่ 2: preprocess อีกนิด
###############################################################################

#ประเภทของข้อมูลในแต่ละ column
#types = train.dtypes
#corr = train.corr()
#description=train.describe()

#เปลี่ยน card_no และ cst_id ให้เป็น categorical 
#Drop pos_dt, open_dt และ exp_dt
train['card_no'] = train['card_no'].astype('category')
train['cst_id'] = train['cst_id'].astype('category')
train = train.drop(['pos_dt', 'open_dt', 'exp_dt'], axis=1)

test['card_no'] = test['card_no'].astype('category')
test['cst_id'] = test['cst_id'].astype('category')
test = test.drop(['pos_dt', 'open_dt', 'exp_dt'], axis=1)

#card_no และ cst_id เป็น categorical drop ทิ้งไปก่อน
train = train.drop(['card_no', 'cst_id'], axis=1)
test = test.drop(['card_no', 'cst_id'], axis=1)

#แบ่ง target ออกจาก train
target = train['npl_flag']
train = train.drop(['npl_flag'], axis=1)

#ใส่ log บนข้อมูล
train = np.log1p(train)
test = np.log1p(test)

#เติมข้อมูลที่เป็น nan
train = train.fillna(train.mean())
test = test.fillna(train.mean())

#หาค่า skew บนข้อมูล
#เราเลือก skewness ที่ >0.75 เพราะ feature ที่มี normal-like distribution
#skewed_feats = train.apply(lambda x: skew(x.dropna()))
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#ดึง list ของ index ที่เหลือออกมา
#skewed_feats = skewed_feats.index
#select_log_train = np.log1p(train[skewed_feats])
#select_log_test = np.log1p(test[skewed_feats])

#เติมข้อมูลที่เป็น nan
#select_log_train = select_log_train.fillna(select_log_train.mean())
#select_log_test = select_log_test.fillna(select_log_test.mean())



###############################################################################
#DEEP
###############################################################################

from keras import optimizers
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

#train = select_log_train.values
#test = select_log_test.values

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
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(output_dim))
model.add(Activation('sigmoid'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#adam = optimizers.Adam(lr=0.0001)

#optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
#model.fit(train, target, batch_size = 10, epochs = 1000)
model.fit(train, target, batch_size=10, epochs = 1000, validation_split=0.2)

#ทำนายผล
y_pred = model.predict(test)
#แปลงค่าคำตอบให้มีแค่ 0 หรือ 1 เท่านั้น
y_round = np.around(y_pred).astype(int)

#save เป็น .txt
np.savetxt('1.txt', y_round, fmt="%d")