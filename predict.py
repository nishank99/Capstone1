from flask import Flask 
import os 
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split#, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import tree
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import graphviz
import image
from PIL import Image 
from numpy import array
from sklearn.metrics import confusion_matrix
#from sklearn.cross_validation import train_test_split
#from sklearn.pipeline import make_pipeline

#import warnings
#warnings.filterwarnings("ignore")
app = Flask(__name__)
app.static_folder = 'static'
@app.route("/")
def main():
	#reading csv data
    data = pd.read_csv('FlaskApp/customer_loan_details.csv')

    #List Attributes
    #print(list(data.columns))

    #Shape of Data
    #print(data.shape)

    #Checking for any null values for the attributes
    #for _ in data.columns:
    #	print("The number of null values in:{} == {}".format(_, data[_].isnull().sum()))

    #Classifying the predictor and the target
    x=data.iloc[:,1:13].values
    y=data.iloc[:,13].values

    #Preprocessing Of Data(Making Categorical Data By defining Levels)
    le=LabelEncoder()
    y=le.fit_transform(y)
    y=y.reshape(684,1)
    #state
    #CA=3,OH=22,FL=6,NY=21,PA=25,MA=13,TX=29,DC=5,AL=1,OK=23,VA=31,MD=14,RI=26,OR=24,TN=28,MI=15,GA=7
    #MT=17,CO=4,NM=19,UT=30,IA=8,SC=27,KY=12,NV=20,MO=16,IN=10,IL=9,NJ=18,AR=2,AK=0,WI=33,WA=32,KS=11
    x[:,0]=le.fit_transform(x[:,0])
    x[:,0]=x[:,0].reshape(684)
    #gender
    #Male=1,Female=0
    x[:,1]=le.fit_transform(x[:,1])
    x[:,1]=x[:,1].reshape(684)
    #race
    #Non-Coapplicant=4,White=6,Not applicable=5,Asian=1,American=0,Native Hawaian=3,Black African =2
    x[:,3]=le.fit_transform(x[:,3])
    x[:,3]=x[:,3].reshape(684)
    #marital_status
    #Married=1,Single=2,Divorced=0
    x[:,4]=le.fit_transform(x[:,4])
    x[:,4]=x[:,4].reshape(684)
    #occupation
    #NYPD=4,IT=2,Accout=0,Business=1,Manager=3
    x[:,5]=le.fit_transform(x[:,5])
    x[:,5]=x[:,5].reshape(684)
    #loan_type
    #Personal=3,Auto=0,Credit=1,Home=2
    x[:,9]=le.fit_transform(x[:,9])
    x[:,9]=x[:,9].reshape(684)
    #Property
    #Urban=2,Rural=0,SemiUrban=1
    x[:,11]=le.fit_transform(x[:,11])
    x[:,11]=x[:,11].reshape(684)




    #Training and Test Data
    x_train2,x_test2,y_train1,y_test1=train_test_split(x,y,test_size=0.25,random_state=0)
    #pred_var = ['applicantId', 'state', 'gender', 'age', 'race', 'marital_status', 'occupation', 'credit_score',
    # 'income', 'debts', 'loan_type', 'LoanAmount', 'Property']
    #X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['loan_decision_type'], \
    #	test_size=0.25, random_state=42)


    #Implementing Decision Tree Algorithm
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(x_train2,y_train1)
    #input={'state':6,'gender':1,'age':32,'race':6,'marital_staus':2,'occupation':1,'credit_score':720,
    #'income':8679.417,'debts':1000,'loan_type':2,'LoanAmount':141,'Property':2}
   
    input1=[6,1,32,6,2,1,720,8679.417,1000,2,141,2] #approved
    input1= array(input1)
    input1=input1.reshape(1,12)

    
    input2=[25,1,48,0,1,0,490,6571,2000,3,349,1] #denied
    input2= array(input2)
    input2=input2.reshape(1,12)

    input3=[21,1,44,5,2,0,540,6238,2500,3,267,2] #denied
    input3= array(input3)
    input3=input3.reshape(1,12)

    input4=[3,1,36,4,1,4,710,9371.333,2000,3,45,2] #approved
    input4= array(input4)
    input4=input4.reshape(1,12)

    input5=[29,1,19,0,2,2,775,4206.5,1757.85,3,114,0] #approved
    input5= array(input5)
    input5=input5.reshape(1,12)

    result = classifier.predict(input5)

    probability=classifier.predict_proba(input5)

    #dot_data = tree.export_graphviz(classifier, out_file=None) 
    #graph = graphviz.Source(dot_data)
    #graph.render("data")
    #features=data.iloc[:,1:13]
    #target=data.iloc[:,13]
    dot_data = tree.export_graphviz(classifier, out_file=None, 
                         feature_names=['state', 'gender', 'age', 'race', 'marital_status', 'occupation', 'credit_score',
     'income', 'debts', 'loan_type', 'LoanAmount', 'Property'],  
                         class_names='loan_decision_type',  
                         filled=True, rounded=True,  
                         special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format='png'
    #graph.render('dtree_render',view=True)
    
  
    #y_train = y_train.replace({'Y':1, 'N':0}).as_matrix()

    #y_test = y_test.replace({'Y':1, 'N':0}).as_matrix()
    #pipe = make_pipeline(PreProcessing(),tree.DecisionTreeClassifier())
    #print(X_train)
    #print(X_test)
    #print(y_train)
    #print(y_test)
    #png_bytes = graph.pipe(format='png')
    #with open('dtree_render.png','wb') as f:
     #   f.write(png_bytes)
    g = graph.render('dtree_render',view=True)
    img = Image.open('dtree_render.png')
    img.save('C:/Users/DELL/AppData/Local/Programs/Python/Python36-32/FlaskApp/static/images/dtree_render.png')
    # make subfolder
    #dir = url_for('static', filename='images')
    #f not os.path.exists(newdir):
     #os.makedirs(newdir)
    #img.save(os.path.join(newdir,'dtree_render.png'))
    return True

          



    #return url_for('static', filename='dtree_render.png')


if __name__ == "__main__":
	app.run()



