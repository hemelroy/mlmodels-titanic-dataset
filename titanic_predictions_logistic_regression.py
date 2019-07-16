import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import log_loss

#Plot confusion matrix to measure performance 
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="green" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

##############Preprocessing of training data##############
survival_df = pd.read_csv("train.csv")
#Not all columns will not be used. Passenger ID and name are not very helpful for this prediction. Ticket and Cabin columns are missing a lot of values
survival_df = survival_df[['Survived','Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
survival_df['Survived'] = survival_df['Survived'].astype('int')

#'Embarked' locations are each assigned a numeric label
survival_df.Embarked.loc[(survival_df['Embarked'] == 'S')] = 0
survival_df.Embarked.loc[(survival_df['Embarked'] == 'C')] = 1
survival_df.Embarked.loc[(survival_df['Embarked'] == 'Q')] = 2

#'Sex' is given a binary label
survival_df.Sex.loc[(survival_df['Sex'] == 'male')] = 0
survival_df.Sex.loc[(survival_df['Sex'] == 'female')] = 1

#Results of assess_missing_data() show 177 missing values for Age. For this model specifically, they have been dropped
survival_df = survival_df[pd.notnull(survival_df['Age'])].copy()

#As the embarked column was only missing 2 values, I included the data for those rows and took the median for the column value
survival_df.Embarked.fillna(survival_df.Embarked.median(),inplace=True)

X = np.asarray(survival_df[['Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])
y = np.asarray(survival_df['Survived'])

#Normalize data set
X = preprocessing.StandardScaler().fit(X).transform(X)

#Split the training data randomly so that 15% of it will be used for testing accuracy
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.15, random_state=432)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


##############Training and Perfomance analysis##############
LR = LogisticRegression(C=0.015, solver='liblinear').fit(X_train,y_train)

print(LR)

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

print("Jaccard Index: ", jaccard_similarity_score(y_test, yhat))
    
print("Confusion Matrix:\n", confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Survived=1','Survived=0'],normalize= False,  title='Confusion matrix')
#plt.show()

print (classification_report(y_test, yhat))

print("Log loss: ", log_loss(y_test, yhat_prob))



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

X1 = np.asarray(df_predict[['Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])

X1[np.isnan(X1)] = 9999

#Normalize data set
X1 = preprocessing.StandardScaler().fit(X1).transform(X1)

y_out = LR.predict(X1)

se = pd.Series(y_out)
df_output['Survived'] = se.values

df_output.to_csv('logregoutput.csv', index=False)