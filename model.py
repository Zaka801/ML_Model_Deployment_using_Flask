import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
df=pandas.read_csv('play1.csv')

features=['OUTLOOK','TEMPERATURE','HUMIDITY','WIND']
x=df[features]
y=df['PLAY']
x_test,x_train,y_test,y_train = train_test_split(x, y , test_size=0.1)
dtree=DecisionTreeClassifier()
dtree=dtree.fit(x,y)
y_pred = dtree.predict(x_test)
print("accuracy",accuracy_score(y_test,y_pred))

pickle.dump(dtree,open("model.pkl","wb"))