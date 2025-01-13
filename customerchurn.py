import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
df = pd.read_csv('customer_data.csv')
df.drop('customerID',axis='columns',inplace=True)

classes = ['customerID','gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn']


df = df[df['TotalCharges'] != ' ']
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

def unique_print_object(df):
    for cols in df:
        if df[cols].dtype == object:
            print(f'{cols} : {df[cols].unique()}')

def unique_print_all(df):
    for cols in df:
        print(f"{cols} : {df[cols].unique()}")

df.replace('No phone service','No',inplace=True)
df.replace('No internet service','No',inplace=True)


cols_with_yes_no = [
    'Partner','Dependents','PhoneService',
    'MultipleLines','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV',
    'StreamingMovies','PaperlessBilling','Churn'
                    ]

for i in cols_with_yes_no:
    df[i].replace('Yes',1,inplace=True)
    df[i].replace('No',0,inplace=True)

for cols in df:
    if df[cols].dtype == object:
        df[cols].fillna(df[cols].mode(),inplace=True)
    else:
        df[cols].fillna(df[cols].mean(),inplace=True)

encoder = LabelEncoder()
df['InternetService'] = encoder.fit_transform(df['InternetService'])
df['Contract'] = encoder.fit_transform(df['Contract'])
df['PaymentMethod'] = encoder.fit_transform(df['PaymentMethod'])
df['gender'] = encoder.fit_transform(df['gender'])

scaler = MinMaxScaler()
df['MonthlyCharges'] = scaler.fit_transform(df[['MonthlyCharges']])
df['TotalCharges'] = scaler.fit_transform(df[['TotalCharges']])
df['tenure'] = scaler.fit_transform(df[['tenure']])

unique_print_all(df)
x = df.drop('Churn',axis='columns')
y = df['Churn']

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=1)
print(np.shape(x_train))
print(np.shape(x_test))

model = keras.Sequential()
model.add(keras.layers.Dense(19,input_shape=(19,),activation="relu"))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(200,activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(50,activation="relu"))
model.add(keras.layers.Dense(25,activation="relu"))
model.add(keras.layers.Dense(1,activation="relu"))
model.compile(optimizer = 'nadam',loss='binary_crossentropy',metrics = ['accuracy'])
model.fit(x_train,y_train,epochs=20)

predicted = model.predict(x_test)

predicted_scaled_data = []
for numbers in predicted:
    if numbers > 0.5:
        numbers = 1
        predicted_scaled_data.append(numbers)
    else:
        numbers = 0
        predicted_scaled_data.append(numbers)
print(model.evaluate(x_test,y_test))

print(y_test[:10])
print(predicted_scaled_data[:10])

print(classification_report(y_test,predicted_scaled_data))

cm = tf.math.confusion_matrix(labels=y_test,predictions=predicted_scaled_data)
plt.figure(figsize=(10,1))
sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
