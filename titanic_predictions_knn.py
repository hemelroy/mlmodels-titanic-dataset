import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Survival: 0 = No, 1 = Yes 

#Check prediction accuracy against various k values and pick the best one 
def assess_k_val(knn_model):
    K_test = 50
    mean_acc = np.zeros((K_test-1))
    std_acc = np.zeros((K_test-1))
    ConfustionMx = [];
    for n in range(1,K_test):

        #Train Model and Predict  
        knn_model = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
        yhat=knn_model.predict(X_test)
        mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)


        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])


    plt.plot(range(1,K_test),mean_acc,'g')
    plt.fill_between(range(1,K_test),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
    plt.legend(('Accuracy ', '+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('K Values (Number of Neighbours)')
    plt.tight_layout()
    plt.show()
    
#Summarize details on missing column values 
def assess_missing_data(dataset):
    print(dataset.isnull().sum())
    

##############Preprocessing of training data##############
df = pd.read_csv('train.csv')

#These columns will not be used. Passenger ID and name are not very helpful for this prediction. Ticket and Cabin columns are missing a lot of values
df = df.drop('PassengerId',1).drop('Name',1).drop('Ticket',1).drop('Cabin',1)

#'Embarked' locations are each assigned a numeric label
df.Embarked.loc[(df['Embarked'] == 'S')] = 0
df.Embarked.loc[(df['Embarked'] == 'C')] = 1
df.Embarked.loc[(df['Embarked'] == 'Q')] = 2

#'Sex' is given a binary label
df.Sex.loc[(df['Sex'] == 'male')] = 0
df.Sex.loc[(df['Sex'] == 'female')] = 1

#Results of assess_missing_data() show 177 missing values for Age. As such, they are assigned a large outlier value to minimize their impact
assess_missing_data(df)
df['Age'] = df['Age'].fillna(value=-9999)

#As the embarked column was only missing 2 values, I included the data for those rows and took the median for the column value
df.Embarked.fillna(df.Embarked.median(),inplace=True)

X = df[['Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values 
y = df['Survived'].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

#Split the training data randomly so that 15% of it will be used for testing accuracy
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.15, random_state=435678)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


##############Training##############
k = 32 #This value is changed based on output of assess_k_val()
#Max happened to be around k=32, model trained with this k value and Manhattan distance
knn_model = KNeighborsClassifier(n_neighbors = k, p = 1).fit(X_train,y_train)
#Euclidean distance: symmetric, spherical, treats all dimensions equally. It is sensitive to extreme differences in single attribute 
#Hamming (for categorical attributes): looks at each attribute and says are they equal or not? If they are, it gets a 1 if not it gets a 0
#Minowski distance: generalizations of euclidean distance. Parameter p dictates what it is. p=2 is Euclidian, p=1 is Manhattan, p=0 is Hamming, p=inf is a max
print(knn_model)

yhat = knn_model.predict(X_test)

print("Train data - Training set's accuracy: ", metrics.accuracy_score(y_train, knn_model.predict(X_train)))
print("Train data - Testing set's accuracy: ", metrics.accuracy_score(y_test, yhat))


assess_k_val(knn_model)


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

X1 = preprocessing.StandardScaler().fit(X1).transform(X1.astype(float))

y_out = knn_model.predict(X1)
print(len(y_out))

se = pd.Series(y_out)
df_output['Survived'] = se.values

df_output.to_csv('knn-predictions.csv', index=False)

