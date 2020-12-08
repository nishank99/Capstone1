from flask import Flask, render_template, json, request,redirect,url_for,jsonify
from flask_mysqldb import MySQL
import MySQLdb
import os 
import json
import numpy as np
import pandas as pd
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split#, GridSearchCV
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import tree
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from numpy import array 
import io
import graphviz
import image
from PIL import Image
#from werkzeug.utils import secure_filename
#UPLOAD_FOLDER = 'C:/Users/DELL/AppData/Local/Programs/Python/Python36-32/FlaskApp/static/data'

app = Flask(__name__)
#app.config['MYSQL_HOST'] = 'localhost'
#app.config['MYSQL_USER'] = 'root'
#app.config['MYSQL_PASSWORD'] = ''
#app.config['MYSQL_DB'] = 'ci'
#mysql = MySQL(app)
app = Flask(__name__)
app.static_folder = 'static'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/")
def main():
    return render_template("index.html")
@app.route("/showLender")
def showLender():
    return render_template("lender.html")
@app.route("/showBorrower")
def showBorrower():
    return render_template("borrower.html")
@app.route("/showSignUp")
def showSignUp():
    return render_template("signup.html")
@app.route("/showResult")
def showResult(prediction):
    return render_template("gallery.html",result=prediction)    

@app.route('/signUp',methods=["POST"])
def signUp():
    # create user code will be here !!
    # read the posted values from the UI
    #cur = mysql.connection.cursor()
    conn = MySQLdb.connect(host="localhost",user="root",password="",db="loan_prediction")
    _name = request.form['name']
    _email = request.form['email']
    _password = request.form['password']

    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (Name,Email,Password)VALUES(%s,%s,%s)",(_name,_email,_password))
    conn.commit()
    return redirect(url_for("showLogin"))

    # validate the received values
    #if _name and _phone and _email and _password:
     #   return json.dumps({'html':'<span>All fields good !!</span>'})
    #else:
     #   return json.dumps({'html':'<span>Enter the required fields</span>'})
    
    #cur.execute('''INSERT INTO users (Name,Phone,Email,Password) VALUES (%s,%s,%s,%s)''',(_name,_phone,_email,_password))
    #query = "INSERT INTO users (Name,Phone,Email,Password) VALUES (_name,_phone,_email,_password)"
    #mysql.connection.commit(query)
    #return mysql.connection.commit()
    #cur = mysql.connection.cursor()
    #cur.execute('''SELECT Name FROM user WHERE Id = 5''')
    #rv = cur.fetchall()
    #return str(rv)
    #return "Done"
    #return redirect(url_for('showSignUp'))
    #return redirect(url_for('correlation_result', security1=security1, 
     #                           security2=security2, start_date=start_date, 
      #                          end_date=end_date))
@app.route("/predict",methods=['GET','POST'])
def predict():

    #reading csv data
    data = pd.read_csv('FlaskApp/customer_loan_details.csv')

    #List Attributes
    print(list(data.columns))

    #Shape of Data
    print(data.shape)

    #Checking for any null values for the attributes
    for _ in data.columns:
       print("The number of null values in:{} == {}".format(_, data[_].isnull().sum()))

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
    train_feature,test_feature,train_target,test_target=train_test_split(x,y,test_size=0.25,random_state=0)
    #pred_var = ['applicantId', 'state', 'gender', 'age', 'race', 'marital_status', 'occupation', 'credit_score',
    # 'income', 'debts', 'loan_type', 'LoanAmount', 'Property']
    #X_train, X_test, y_train, y_test = train_test_split(data[pred_var], data['loan_decision_type'], \
    #   test_size=0.25, random_state=42)


    #Implementing Decision Tree Algorithm
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(train_feature,train_target)
    #target_pred=classifier.predict(test_feature)
    #cm=confusion_matrix(test_target,target_pred)

    #input={'state':6,'gender':1,'age':32,'race':6,'marital_staus':2,'occupation':1,'credit_score':720,
    #'income':8679.417,'debts':1000,'loan_type':2,'LoanAmount':141,'Property':2}
   
    #input1=[6,1,32,6,2,1,720,8679.417,1000,2,141,2]
    #input1= le.fit_transform(input1)
    #input1=input1.reshape(1,12)

    #input2=[21,1,44,5,2,0,540,6238,2500,3,267,2]
    #input2= le.fit_transform(input2)
    #input2=input2.reshape(1,12)

    #user input
    _state = request.form['state']
    _gender = request.form['gender']
    _age = int(request.form['age'])
    _race = request.form['race'] 
    _marital_status = request.form['marital_status'] 
    _occupation = request.form['occupation'] 
    _credit_score = int(request.form['credit_score'])
    _income = float(request.form['income'])
    _debts = float(request.form['debts'])
    _loan_type = request.form['loan_type']
    _LoanAmount = int(request.form['LoanAmount'])
    _Property = request.form['Property']
    
    #categorical data
    select_state = {'CA':3,'OH':22,'FL':6,'NY':21,'PA':25,'MA':13,'TX':29,'DC':5,'AL':1,'OK':23,'VA':31,
    'MD':14,'RI':26,'OR':24,'TN':28,'MI':15,'GA':7,'MT':17,'CO':4,'NM':19,'UT':30,'IA':8,'SC':27,
    'KY':12,'NV':20,'MO':16,'IN':10,'IL':9,'NJ':18,'AR':2,'AK':0,'WI':33,'WA':32,'KS':11}
    _state=select_state.get(_state,"Invalid state")

    select_gender = {'Male':1,'Female':0}
    _gender=select_gender.get(_gender,"Invalid gender")
  
    select_race = {'Non-Coapplicant':4,'White':6,'Not applicable':5,'Asian':1,'American':0,
    'Native Hawaian':3,'Black African':2}
    _race=select_race.get(_race,"Invalid race")

    select_ms = {'Married':1,'Single':2,'Divorced':0}
    _marital_status=select_ms.get(_marital_status,"Invalid marital_status")

    select_occupation = {'NYPD':4,'IT':2,'Accout':0,'Business':1,'Manager':3}
    _occupation=select_occupation.get(_occupation,"Invalid occupation")

    select_loan_type = {'Personal':3,'Auto':0,'Credit':1,'Home':2}
    _loan_type=select_loan_type.get(_loan_type,"Invalid loan_type")

    select_property = {'Urban':2,'Rural':0,'SemiUrban':1}
    _Property=select_property.get(_Property,"Invalid property")

    user_input = [_state,_gender,_age,_race,_marital_status,_occupation,_credit_score,_income,_debts,
    _loan_type,_LoanAmount,_Property]

    user_input=array(user_input)
    user_input=user_input.reshape(1,12)

    result = classifier.predict(user_input)

    if result==0:
        prediction="Yipee!Your Loan Will Get Approved!"
    else:
        prediction="Oops!Your Loan Will Not Get Approved"

    probability=classifier.predict_proba(user_input)
      
    return showResult(prediction)
    
@app.route("/showLogin")
def showLogin():
    return render_template("login.html")
@app.route("/login",methods=["POST"])
def login():
    conn = MySQLdb.connect(host="localhost",user="root",password="",db="loan_prediction")
    email = str(request.form["email"])
    password = str(request.form["password"])
    cursor = conn.cursor()
    cursor.execute("SELECT Email FROM users WHERE Email ='"+email+"' AND Password ='"+password+"'")
    user = cursor.fetchone()

    if user is None or len(user) == 0:
                return "failed"
                
    if len(user) is 1:
        return redirect(url_for('main'))
    else:
        return "failed"
    return render_template("login.html")
@app.route("/showAbout")
def showAbout():
    return render_template("about.html")
@app.route("/showContact")
def showContact():
    return render_template("contact.html")
@app.route("/lend",methods=['POST'])
def lend():
    csv_file = request.files['data']
    #stream = io.StringIO(csv_file.stream.read().decode("UTF8"), newline=None)
    data = pd.read_csv(csv_file)
    #filename = secure_filename(csv_file.filename)
    #csv_input.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #data = pd.read_csv('C:/Users/DELL/AppData/Local/Programs/Python/Python36-32/FlaskApp/static/data/'+filename)

    #List Attributes
    print(list(data.columns))

    #Shape of Data
    print(data.shape)

    #Checking for any null values for the attributes
    for _ in data.columns:
       print("The number of null values in:{} == {}".format(_, data[_].isnull().sum()))

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
    train_feature,test_feature,train_target,test_target=train_test_split(x,y,test_size=0.25,random_state=0)

    #Implementing Decision Tree Algorithm
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(train_feature,train_target)

    dot_data = tree.export_graphviz(classifier, out_file=None, 
                         feature_names=['state', 'gender', 'age', 'race', 'marital_status', 'occupation', 'credit_score',
     'income', 'debts', 'loan_type', 'LoanAmount', 'Property'],  
                         class_names='loan_decision_type',  
                         filled=True, rounded=True,  
                         special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format='png'
    graph.render('dtree_render',view=True)
    img = Image.open('dtree_render.png')
    img.save('C:/Users/DELL/AppData/Local/Programs/Python/Python36-32/FlaskApp/static/images/dtree_render.png')



    result = classifier.predict(test_feature)
    for i in result:
        if i==0:
            i="Approved!"
        else:
            i="Denied"
    return str(result)    


if __name__ == "__main__":
	app.run()