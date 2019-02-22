import numpy as np;
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pandas as pd
import tensorflow as tf
from tensorflow import  keras

fileRows = pd.read_csv("Gender.csv")

names =[]
gender =[]

for name in fileRows['name']:
    x = []
    for code in bytearray(name.lower(),'ascii'):
       x.append(code)

    if len(x) < 19:
        for i in range(19 - len(x)):
            x.append(0)

    names.append(x)


for gen in fileRows['gender']:
    gender.append(int(gen))

test_data = []



names = np.array(names)
names = names[0:,0:7]
#print(names)

#print(gender)
for code in bytearray('seema`','ascii'):
    x = []
    x.append(code)
    if len(x) < 7:
       for i in range(7 - len(x)):
           x.append(0)

test_data.append(x);
test_data = np.array(test_data)

"""
clf = svm.SVR(gamma='auto')
clf.fit(names,gender)
c_predict = clf.predict(test_data)
print(c_predict)
"""

#Support Vector Classifier
s_clf = SVC(gamma='auto')
#s_clf.fit(names,gender)
#s_prediction = s_clf.predict(test_data)
#print(s_prediction)

names = names.reshape((28918, 7,1 ))
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(7,1)),
    keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

test_data = test_data.reshape(1,7,1)
model.fit(names, gender, epochs=5)
predictions = model.predict(test_data)
print(np.argmax(predictions[0]))