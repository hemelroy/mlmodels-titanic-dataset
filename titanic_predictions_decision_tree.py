import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import graphviz
import matplotlib.image as mpimg
from sklearn import tree

##############Preprocessing of training data##############
train_df = pd.read_csv("train.csv", delimiter=",")

train_df.Embarked.loc[(train_df['Embarked'] == 'S')] = 0
train_df.Embarked.loc[(train_df['Embarked'] == 'C')] = 1
train_df.Embarked.loc[(train_df['Embarked'] == 'Q')] = 2

#There are 177 missing values for Age. As such, they have been dropped for this specific algorithm
train_df = train_df[pd.notnull(train_df['Age'])].copy()

#As the embarked column was only missing 2 values, I included the data for those rows and took the median for the column value
train_df.Embarked.fillna(train_df.Embarked.median(),inplace=True)

train_df.Sex.loc[(train_df['Sex'] == 'male')] = 0
train_df.Sex.loc[(train_df['Sex'] == 'female')] = 1

X = train_df[['Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values

y = train_df["Survived"]

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.2, random_state=3)

##############Training and performance measurement##############
tree_model = DecisionTreeClassifier(criterion="entropy", max_depth = 6)
print(tree_model) 

tree_model.fit(X_trainset,y_trainset)

predictions = tree_model.predict(X_testset)
print(predictions)

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predictions))

#Visualization of decision tree (refer to png image in repo)
dot_data = StringIO()

df = train_df['Embarked'].to_string()
print(np.unique(y_trainset))

filename = "survivaltree.png"
featureNames = ['Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
targetNames = train_df["Survived"].unique().tolist()
out=tree.export_graphviz(tree_model,feature_names=featureNames, out_file=dot_data, class_names=['0','1'], filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png(filename)
#img = mpimg.imread(filename)
#plt.figure(figsize=(100, 200))
#plt.imshow(img,interpolation='nearest')


##############Make predictions on the actual data using the model created##############
#Cleaning up the data into data frames 
df_predict = pd.read_csv('test.csv')
df_output = pd.DataFrame(columns = ['PassengerId', 'Survived'])
df_output['PassengerId'] = df_predict['PassengerId'].astype(str)
df_predict = df_predict.drop('PassengerId',1).drop('Name',1).drop('Ticket',1).drop('Cabin',1)

df_predict.Embarked.loc[(df_predict['Embarked'] == 'S')] = 0
df_predict.Embarked.loc[(df_predict['Embarked'] == 'C')] = 1
df_predict.Embarked.loc[(df_predict['Embarked'] == 'Q')] = 2

df_predict.Sex.loc[(df_predict['Sex'] == 'male')] = 0
df_predict.Sex.loc[(df_predict['Sex'] == 'female')] = 1

X1 = df_predict[['Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values

#Unspecified values are assigned an outlier value of 9999
X1[np.isnan(X1)] = 9999

print(X1)

y_out = tree_model.predict(X1)
print(len(y_out))

se = pd.Series(y_out)
df_output['Survived'] = se.values

df_output.to_csv('decisiontree.csv', index=False)