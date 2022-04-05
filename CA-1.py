

from statistics import median
from tkinter.font import BOLD
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

dataset = pd.read_csv('diabetes.csv')
n_samples , n_features = dataset.shape

#Cleaning Data 

#Checking the datatypes of each columns
print("-------------Data Types----------------")
print(dataset.dtypes)

#Checking whether any data is null
print("\n-------------No Of datas that are null----------------\n")
print(dataset.isnull().sum(axis=0))

#Checking for incorrect data and invalid zero in data
print("\n-------------Percentage of Missing Values----------------\n")
print(((dataset[:] == 0).sum())/n_samples*100)

#Plot histogram to understand the distribution of data
dataset.hist(figsize=(10,10))
plt.show()

#As percentage of missing values in some features(BloodPressure, BMI, Glucose) are very less
#Remove those data 

dataset.drop(dataset[(dataset['BloodPressure'] == 0 )].index, inplace=True)
dataset.drop(dataset[(dataset['BMI'] == 0 )].index, inplace=True)
dataset.drop(dataset[(dataset['Glucose'] == 0 )].index, inplace=True)

#As percentage of missing values in features like Insulin 
#and skin thickness are comparitively high
#Those datas are replaced with median of the existing data

insulinMedian = dataset['Insulin'].median()
thicknessMedian = dataset['SkinThickness'].median()

dataset['Insulin'].replace(0,np.nan, inplace= True)
dataset['Insulin'].fillna(insulinMedian, inplace= True)

dataset['SkinThickness'].replace(0,np.nan, inplace= True)
dataset['SkinThickness'].fillna(thicknessMedian, inplace= True)

print("\n---------------Sample size after Cleaning--------------------------\n")
n_samples , n_features = dataset.shape
print(n_samples,n_features)

print("\n---------------Percentage of Missing Values after Cleaning----------------\n")
print(((dataset[:] == 0).sum())/n_samples*100)

#Feature Selection 
print(dataset.head())
pd.set_option('display.width', 100)
pd.set_option('display.precision', 3)
# dataset.rename(columns={'DiabetesPedigreeFunction': 'pedigree'})
print(dataset.describe())

#checking distribution of outcome
f, ax = plt.subplots(figsize= (7,5))
sns.countplot(x='Outcome', data=dataset)
plt.title('# Chances to have diabetes vs No chances')
plt.xlabel('Class (1==Chances to have diabetes)')
plt.show()

#checking correlation
corr_df = dataset.corr()
print("\n------------------------Correlation-----------------\n")
print(corr_df)

#Plotting heatmap to visualize correlation
plt.figure(figsize = (10,6))
ax = plt.axes()
sns.heatmap(corr_df, ax = ax)
ax.set_title('Correlation between features')
plt.show()

corr_df_viz = corr_df
corr_df_viz['feature'] = corr_df_viz.index
plt.figure(figsize = (10,6))

#create bargraph to select top features
sns.barplot(x= 'feature', y = "Outcome", data = corr_df_viz, order= corr_df_viz.sort_values('Outcome', ascending = False).feature)

#set labels
plt.xlabel("Feature", size=15)
plt.ylabel("Correlation Between chances to have diabetes", size= 15)
plt.title("Arranging Features on Priority", size = 20)
plt.tight_layout()
plt.xticks(rotation = 80)
plt.show()

"""Top 3 features are Glucose, BMI, age"""

# Feature Selection with Univariate Statistical Tests
# feature extraction
X = dataset.drop(['Outcome'], axis = 1)
Y = dataset['Outcome']
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print("\nFit Scores are", fit.scores_)
features = fit.transform(X)
# summarize selected features
print("Features are", features[0:5,:])

"""The attributes with the highest scores are to be chosen"""
"""selected features are Glucose, Age, BMI"""

dataset.drop(['Pregnancies','BloodPressure','SkinThickness', 'Insulin','DiabetesPedigreeFunction'], axis = 1, inplace= True)
n_samples , n_features = dataset.shape
print("\n",dataset.head())

X = dataset.iloc[:, :n_features-1]
Y = dataset.iloc[:,n_features-1]

# Splitting the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)
classLabels = ['Non-Diabetic','Diabetic']

#Implementing RandomForest
print("\n---------------------Random Forest Classifier-----------------------\n")
randomForest = RandomForestClassifier(max_depth= 5, random_state= 0)
randomForest.fit(x_train,y_train)
y_predRF = randomForest.predict(x_test)

#Calculating Test accuracy,confusion matrix and classification report
matrix = confusion_matrix(y_test, y_predRF)
print("Confusion matrix of Random Forest\n", matrix)
cmPlot1 = ConfusionMatrixDisplay(matrix,display_labels= classLabels)
accuracyRF = accuracy_score(y_test,y_predRF)
print("\nAccuracy : %0.2f" % (accuracyRF*100) , "%")
print("Precision: %0.2f" % precision_score(y_test,y_predRF))
print("Recall: %0.2f" % recall_score(y_test,y_predRF))
print("f1-score: %0.2f" % f1_score(y_test,y_predRF))

#Implementing KNN
print("---------------------KNN-----------------------\n")
knn = KNeighborsClassifier(n_neighbors= 11)
knn.fit(x_train,y_train)
y_PredKNN = knn.predict(x_test)

#Calculating Test accuracy, confusion matrix 
matrix = confusion_matrix(y_test, y_PredKNN)
print("Confusion matrix of KNN\n", matrix)
cmPlot2 = ConfusionMatrixDisplay(matrix,display_labels= classLabels)

print("\nAccuracy: %0.2f" % (accuracy_score(y_test,y_PredKNN) * 100),"%")
print("Presicion : %0.2f" % precision_score(y_test,y_PredKNN))
print("Recall: %0.2f" % recall_score(y_test,y_PredKNN))
print("f1-score: %0.2f" % f1_score(y_test,y_PredKNN))

#Implementing SVC
print("---------------------SVC-----------------------")
scaler = StandardScaler()
model_svc = Pipeline([('standardize', scaler), ('log_read', SVC(probability = True))])
model_svc.fit(x_train, y_train)
model_svc.score(x_test, y_test)
Pipeline( steps =[('standardize', StandardScaler()),
                               ('svc', SVC())],)

# Accuracy for Test Split
y_test_h = model_svc.predict(x_test)
y_test_h_probs = model_svc.predict_proba(x_test)[:,1]
matrix = confusion_matrix(y_test, y_test_h)
print('\nConfusion matrix of SVM:\n', matrix)
cmPlot3 = ConfusionMatrixDisplay(matrix,display_labels = classLabels)
test_accuracy = accuracy_score(y_test, y_test_h)*100
print('Accuracy: %.2f %%' % test_accuracy)
print("Presicion: %0.2f" % precision_score(y_test, y_test_h))
print("Recall: %0.2f" % recall_score(y_test,y_test_h))
print("f1-score: %0.2f" % f1_score(y_test,y_test_h))

#Plotting Confusion Matrix
f, axes = plt.subplots(1,3, sharey='row', figsize = (10,10))

cmPlot1.plot(ax=axes[0], xticks_rotation=45)
cmPlot1.ax_.set_title('Confusion matrix of Random Forest', size = 8)
cmPlot1.im_.colorbar.remove()
cmPlot1.ax_.set_xlabel('')

cmPlot2.plot(ax=axes[1], xticks_rotation=45)
cmPlot2.ax_.set_title('Confusion matrix of KNN' , size = 8)
cmPlot2.im_.colorbar.remove()
cmPlot2.ax_.set_xlabel('')
cmPlot2.ax_.set_ylabel('')

cmPlot3.plot(ax=axes[2], xticks_rotation=45)
cmPlot3.ax_.set_title('\n \nConfusion matrix of SVM', size = 8)
cmPlot3.im_.colorbar.remove()
cmPlot3.ax_.set_xlabel('')
cmPlot3.ax_.set_ylabel('')

f.text(0.5, 0.2, 'Predicted label', ha='center')
plt.subplots_adjust(hspace=0.3)
plt.show()
