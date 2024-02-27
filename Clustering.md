# ANALYSING STUDENT'S BEHAVIOUR

## Importing necessary libraries
The required libraries such as pandas,numpy,seaborn,mathplotlib etc are imported.
NB:Install the libraries if not present,using pip install corresponding library name.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

## Loading the data to be analysed
The dataset can be in the form of csv file or may be directly from another sites.Specify the path of the dataset that we need to upload.

```python
df_class=pd.read_csv("Student behaviour.csv")
df_class.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/9a8a6581-94a7-4c70-8ba2-e7afa0a0fc19)


## Optimizing the dataset
In pandas, the info() method is used to get a concise summary of a DataFrame. This method provides information about the DataFrame, including the data types of each column, the number of non-null values, and memory usage. It's a handy tool for quickly assessing the structure and content of your DataFrame.
```python
df_class.info()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/08d36197-3138-45e8-84bd-2f9f11a80f76)

## Rremoving unnecessary columns
```python
df_class= df_class.drop(['NationalITy','PlaceofBirth','StageID','GradeID','SectionID','Topic','Semester','Relation'],axis=1)
```

## Exploratory Data Analysis
Relation between Raising Hands and Classification
```python
df_class['raisedhands'] = pd.cut(df_class.raisedhands, bins=3, labels=np.arange(3), right=False)
df_class.groupby(['raisedhands'])['Class'].value_counts(normalize=True)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/4d241ca5-8d18-42ac-bace-e63025c6891b)


Relation between Visited Resourses and Classification

```python
df_class['VisITedResources'] = pd.cut(df_class.VisITedResources, bins=3, labels=np.arange(3), right=False)
df_class.groupby(['VisITedResources'])['Class'].value_counts(normalize=True)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/82c4b6dd-fea4-422c-8c84-43247dc4f7d7)


Relation between Announcements View  and Classification
```python
df_class['AnnouncementsView'] = pd.cut(df_class.AnnouncementsView, bins=3, labels=np.arange(3), right=False)
df_class.groupby(['AnnouncementsView'])['Class'].value_counts(normalize=True)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/a4cd0ca7-08e0-45f2-bd5d-8227e1a4296e)

Relation between Discussion  and Classification
```python
df_class['Discussion'] = pd.cut(df_class.Discussion, bins=3, labels=np.arange(3), right=False)
df_class.groupby(['Discussion'])['Class'].value_counts(normalize=True)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/e34a8c1c-0990-4732-af3a-5243b6934c1b)

Relation between StudentAbsenceDays and Classification
```python
df_class.groupby(['StudentAbsenceDays'])['Class'].value_counts(normalize=True)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/f69128af-0c05-4885-b9be-854b576b3274)

## Visualization

1) Creating boxplot on raisedhands v/s Class
```python
sns.boxplot(y=df_class['Class'],x=df_class['raisedhands'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/b508cbc0-eb65-4bed-b0f4-f16294a15b21)


2) Creating boxplot on VisITedResources v/s Class
```python
sns.boxplot(y=df_class['Class'],x=df_class['VisITedResources'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/bf652135-4019-41af-9b7c-fcafe56b51d5)


3) Creating boxplot on AnnouncementsView v/s Class
```python
sns.boxplot(y=df_class['Class'],x=df_class['AnnouncementsView'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/f741d445-55da-44bd-8202-e38f05375ea2)


4) Creating boxplot on Discussion v/s Class
```python
sns.boxplot(y=df_class['Class'],x=df_class['Discussion'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/f543c926-6642-43b2-b40f-636985d0082f)


5) Creating boxplot on StudentAbsenceDays v/s Class
```python
sns.boxplot(y=df_class['Class'],x=df_class['StudentAbsenceDays'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/709dca46-7645-4783-9267-362bc9703537)

## Correlation 
```python
correlation = df_class[['raisedhands','VisITedResources','AnnouncementsView','Discussion']].corr(method='pearson')
correlation
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/d450a1f8-b8e3-4771-90f5-208dae696609)


## Elbow Method
```python
X = df_class[['raisedhands', 'VisITedResources']].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    #print (i,kmeans.inertia_)
    wcss.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss,marker='o')
plt.title('Elbow Method')
plt.xlabel('N of Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/3bf23206-160e-4fd3-9355-0b3f022289ad)


## K-Means Clustering
```python
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)

k_means_labels = kmeans.labels_
k_means_cluster_centers = kmeans.cluster_centers_

plt.scatter(X[:, 0], X[:,1], s = 10, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'red',label = 'Centroids')
plt.title('Students Clustering')
plt.xlabel('RaisedHands')
plt.ylabel('VisITedResources')
plt.legend()
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/8062df5b-6cab-41a3-b03f-38ed6c3718de)
