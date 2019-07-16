import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score

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
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
##############Preprocessing of training data##############
df = pd.read_csv("train.csv")
df.Embarked.loc[(df['Embarked'] == 'S')] = 0
df.Embarked.loc[(df['Embarked'] == 'C')] = 1
df.Embarked.loc[(df['Embarked'] == 'Q')] = 2

#There are 177 missing values for Age. As such, they have been dropped for this specific algorithm
df = df[pd.notnull(df['Age'])].copy()

#As the embarked column was only missing 2 values, I included the data for those rows and took the median for the column value
df.Embarked.fillna(df.Embarked.median(),inplace=True)

df.Sex.loc[(df['Sex'] == 'male')] = 0
df.Sex.loc[(df['Sex'] == 'female')] = 1

survival_df = df[['Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X = np.asarray(survival_df)

df['Survived'] = df['Survived'].astype('int')
y = np.asarray(df['Survived'])

#Split the training data randomly so that 20% of it will be used for testing accuracy
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=34567)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


##############Training and performance measurement##############
classifier = svm.SVC(kernel='rbf', gamma='auto')
classifier.fit(X_train, y_train) 

yhat = classifier.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['No Survive','Survive'],normalize= False,  title='Confusion matrix')


f1 = f1_score(y_test, yhat, average='weighted') 
jaccard = jaccard_similarity_score(y_test, yhat)

print("F1 score", f1, "Jaccard:", jaccard)

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
y1 = np.asarray(df['Survived'])

X1[np.isnan(X1)] = 9999

y_out = classifier.predict(X1)

se = pd.Series(y_out)
df_output['Survived'] = se.values

df_output.to_csv('svmoutput.csv', index=False)



