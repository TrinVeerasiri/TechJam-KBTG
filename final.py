###############################################################################
#ขั้นที่ 1: import และโหลดข้อมูล
###############################################################################

#import ข้อมูล (ข้อมูลนี้ใส่ log มาแล้ว)
import math
import pandas as pd
import numpy as np
from scipy.stats import skew

master_card_no = pd.read_csv('master_card_no.txt')
test = pd.read_csv('test_set.txt')
train = pd.read_csv('train_set.txt')

train = train.drop(['card_no'], axis=1)
test = test.drop(['card_no'], axis=1)

train = train.fillna(0)
test = test.fillna(0)

train['Categories'] = train['Categories'].astype('category')
train['card_type'] = train['card_type'].astype('category')
train['zip_province'] = train['zip_province'].astype('category')
train['first_card'] = train['first_card'].astype('category')
train['gender'] = train['gender'].astype('category')

#for amount drop y_amt that equal to 0
#train = train[train.y_amt != 0]

test['Categories'] = test['Categories'].astype('category')
test['card_type'] = test['card_type'].astype('category')
test['zip_province'] = test['zip_province'].astype('category')
test['first_card'] = test['first_card'].astype('category')
test['gender'] = test['gender'].astype('category')

#train y_n first
target_y_n = train['y_n']
target_y_amt = train['y_amt']

#drop target from train set
train = train.drop(['y_n', 'y_amt'], axis=1)

#target_y_amt = np.log1p(target_y_amt)


###############################################################################
#DEEP
###############################################################################

from keras import optimizers
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

train = train.values
test = test.values
#target_y_amt = target_y_amt.values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
train[:, 0] = labelencoder_X_1.fit_transform(train[:, 0])
test[:, 0] = labelencoder_X_1.fit_transform(test[:, 0])
labelencoder_X_2 = LabelEncoder()
train[:, 1] = labelencoder_X_2.fit_transform(train[:, 1])
test[:, 1] = labelencoder_X_1.fit_transform(test[:, 1])
labelencoder_X_3 = LabelEncoder()
train[:, 10] = labelencoder_X_3.fit_transform(train[:, 10])
test[:, 10] = labelencoder_X_1.fit_transform(test[:, 10])
labelencoder_X_4 = LabelEncoder()
train[:, 11] = labelencoder_X_4.fit_transform(train[:, 11])
test[:, 11] = labelencoder_X_1.fit_transform(test[:, 11])


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)


input_dim = train.shape[1]
output_dim = 1


#โมเดลของตฤณ
model = Sequential()
model.add(Dense(64, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(output_dim))
model.add(Activation('linear'))
    

#สำหรับจำนวนครั้ง
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#สำหรับจำนวนเงิน
#adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


#adam = optimizers.Adam(lr=0.0001)
RMSprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=1e-6)

#optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mse", optimizer=adam, metrics = ['accuracy'])
#model.fit(train, target, batch_size = 10, epochs = 1000)
model.fit(train, target_y_n, batch_size=200, epochs = 100, validation_split=0.2)





#ทำนายผล
y_pred = model.predict(test)
#แปลงค่าคำตอบให้มีแค่ 0 หรือ 1 เท่านั้น
y_round = np.around(y_pred).astype(int)

y_pred_new = np.floor(y_pred)

#save y_n เป็น .txt
np.savetxt('y_n.txt', y_round, fmt="%d")

#save y_amt เป็น .txt
np.savetxt('y_amt.txt', y_pred, fmt="%d")



###############################################################################
#Ridge regression
###############################################################################

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet,\
     LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

#cv=5 เมื่อเราใส่ rmse_cv(model) จะออกมา 5 ค่า
#เข้าใจว่าต้องใช้ -cross_val_score เพราะ scoring เป็น neg_mean_squared_error
#ไม่งั้นค่าจะติดลบ ไม่สามารถถอด sqrt ได้
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train, target_y_amt, \
                                   scoring="neg_mean_squared_error", cv = 5))
    return(rmse)
    
#Linear least squares with l2 regularization
model_ridge = Ridge()

#ค่า alpha ใช้สำหรับปรับ Regularization พารามิเตอร์ของ Ridge
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
#ตรงนี้ loop ดนพ จะไล่ค่า alpha ไปทีละค่า ซึ่งแต่ละค่า alpha จะทำนายออกมา 5 ครั้ง จึงต้องหา
#ค่า mean
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

#เดิม cv_ridge เป็น list จึงใส่ใน pd.Series() เพื่อใช้คำสั่ง .plot
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

#หาค่า cv_ridge ที่น้อยที่สุด
cv_ridge.min()
