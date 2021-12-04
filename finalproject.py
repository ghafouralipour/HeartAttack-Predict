import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
# RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def load_heart_data(file_address):
    return pd.read_csv(file_address)

def load_saturation(file_address):
    return pd.read_csv(file_address)
def print_data_spec(data):
    print("===================== Data Head =====================")
    print(data.head())
    print("===================== Data Info =====================")
    print(data.info())
    print("===================== Data Coloumns =================")
    print(data.columns)    

def show_coloumn_in_bar(column,x_label,y_label,title):
    plt.rcParams['figure.figsize']=(20,10)
    sns.countplot(heart_data[column])
    plt.xlabel(x_label, size=18)
    plt.ylabel(y_label,size=18)
    plt.title(title, size=25)
    plt.show()
def show_coloumn_hist(column,color,bins,x_label,y_label,title):
    plt.rcParams['figure.figsize']=(20,10)
    plt.hist(heart_data[column], color=[color], bins=bins)
    plt.xlabel(x_label, size=18)
    plt.ylabel(y_label,size=18)
    plt.title(title, size=25)    
    plt.show()

def prevalence_of_heart_attack_bar():
    plt.figure(figsize=(20,30))

    plt.subplot(5,2,1)
    plt.title('Prevalence of Heart attack by Sex',fontsize=15)
    sns.countplot(heart_data['output'], hue=heart_data['sex'])

    plt.subplot(5,2,2)
    plt.title('Prevalence of Heart attack by Chest Pain',fontsize=15)
    sns.countplot(heart_data['output'], hue=heart_data['cp'])

    plt.subplot(5,2,3)
    plt.title('Prevalence of Heart attack by fasting blood sugar > 120 mg/dl',fontsize=15)
    sns.countplot(heart_data['output'],hue=heart_data['fbs'])

    plt.subplot(5,2,4)
    plt.title('Prevalence of Heart attack by restecg',fontsize=15)
    sns.countplot(heart_data['output'],hue = heart_data['restecg'])

    plt.subplot(5,2,5)
    plt.title('Prevalence of Heart attack by Exercise induced angina',fontsize=15)
    sns.countplot(heart_data['output'],hue=heart_data['exng'])

    plt.subplot(5,2,6)
    plt.title('Prevalence of Heart attack by slp',fontsize=15)
    sns.countplot(heart_data['output'],hue=heart_data['slp'])

    plt.subplot(5,2,7)
    plt.title('Prevalence of Heart attack by number of major vessels',fontsize=15)
    sns.countplot(heart_data['output'],hue=heart_data['caa'])

    plt.subplot(5,2,8)
    plt.title('Prevalence of Heart attack by thall',fontsize=15)
    sns.countplot(heart_data['output'],hue=heart_data['thall'])


def prevalence_heart_attack_col(x_coloumn,y_coloumn,title,x_label,plot_kind):
    # output reprsents whether the person had a heart-attack (output=1) or not (output=0)
    data = pd.crosstab(heart_data[x_coloumn], heart_data[y_coloumn])
    data.div(data.sum(1).astype(float),axis=0).plot(kind=plot_kind, stacked=True, figsize=(20,10), color=['blue','pink'])
    plt.title(title, fontsize = 30)
    plt.xlabel(x_label, fontsize = 15)
    plt.legend()
    plt.show()

def normalize_data(heart_data):
    heart_data['age']= heart_data['age']/max(heart_data['age'])
    heart_data['trtbps']= heart_data['trtbps']/max(heart_data['trtbps'])
    heart_data['cp']= heart_data['cp']/max(heart_data['cp'])
    heart_data['chol']= heart_data['chol']/max(heart_data['chol'])
    heart_data['thalachh']= heart_data['thalachh']/max(heart_data['thalachh'])
    return heart_data

def create_test_train_data(heart_data):
    X_train, X_test, y_train, y_test = train_test_split( heart_data.drop(['output'],axis=1), 
                                                        heart_data.output, test_size=0.2, random_state=0,
                                                        stratify = heart_data.output)
    return X_train, X_test, y_train, y_test

def logistic_regression_learn(X_train, X_test, y_train, y_test):
    clf= LogisticRegression()
    params= {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge

    clf_grid= GridSearchCV(estimator= clf, param_grid= params, cv=5)
    clf_grid.fit(X_train,y_train)
    # clf.get_params()
    y_pred= clf_grid.predict(X_test)
    acc= accuracy_score(y_test, y_pred)
    print('Accuracy for Logistic Regression: ', acc)
    
def random_forest_learn(X_train, X_test, y_train, y_test):
    # 
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred2= classifier.predict(X_test)
    acc2= accuracy_score(y_test, y_pred2)
    print('Accuracy for RandomForest Classifier is: ', acc2)

def K_neighbor_learn(X_train, X_test, y_train, y_test):
    classifier=KNeighborsClassifier()

    params1 = {
        'n_neighbors': (1,10, 1),
        'leaf_size': (20,40,1),
        'p': (1,2),
        'weights': ('uniform', 'distance'),
        'metric': ('minkowski', 'chebyshev'),}
    clf2_grid= GridSearchCV(estimator= classifier, param_grid= params1, cv=5)
    clf2_grid.fit(X_train,y_train)
    y_pred3= clf2_grid.predict(X_test)
    acc3= accuracy_score(y_test, y_pred3)
    print('Accuracy for K neighbor classifier is: ', acc3)

if __name__ == "__main__":
    #load datasets
    heart_data_dataset_address= 'Heart Attack prediction/heart.csv'
    heart_data = load_heart_data(heart_data_dataset_address)
    #dataset spec
    print_data_spec(heart_data)
    #data distribution
    show_coloumn_in_bar('age','Age','Count','Age Distribution')
    show_coloumn_in_bar('sex','Sex','Count','Sex Distribution')
    show_coloumn_in_bar('cp','Chest Pain Type','Count','Type of chest pain Distribution')
    bins=(80,100,110,120,130,140,150,160,190)
    show_coloumn_hist('trtbps','pink',bins,'trtbps','Count','trtbps Distribution')
    bins=(100,200,300,400,500)
    show_coloumn_hist('chol','red',bins,'Cholestoral level','Count','cholestoral Distribution')
    # fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
    show_coloumn_in_bar('fbs','fbs','Count','fasting blood sugar Distribution')
    #prevalence
    prevalence_of_heart_attack_bar()
    prevalence_heart_attack_col('age','output','Prevalence of heart attack by age','Age','bar')
    prevalence_heart_attack_col('trtbps','output','Prevalence of heart attack by resting blood pressure','trtbps','bar')
    prevalence_heart_attack_col('chol','output','Prevalence of heart attack by cholestrol','cholestrol','hist')
    prevalence_heart_attack_col('thalachh','output','Prevalence of heart attack by maximum heart rate achieved',
                                'thalachh','hist')
    #machine learning
    heart_data = normalize_data(heart_data)
    X_train, X_test, y_train, y_test = create_test_train_data(heart_data)
    logistic_regression_learn(X_train, X_test, y_train, y_test)
    random_forest_learn(X_train, X_test, y_train, y_test)
    K_neighbor_learn(X_train, X_test, y_train, y_test)