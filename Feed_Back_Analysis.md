# Excersise-5

## Use machine learning techniques to analyse the feedback of the Intel Unnati sessions

### Step-1: Importing necessary libraries
The required libraries such as `pandas,numpy,seaborn,mathplotlib` etc are imported.
NB:Install the libraries if not present,using ``` pip install ``` corresponding library name.
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

### Step-2: Loading the data to be analysed
The dataset can be in the form of csv file or may be directly from another sites.Specify the path of the dataset that we need to upload.
```python
#df_class=pd.read_csv("/content/survey_data.csv")
df_class=pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
```
```python
df_class.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/13f52d0a-1de0-411c-bd8b-89d7e537741d)

To make the table more attractive we can use the following commands:
```python
df_class.sample(5).style.set_properties(**{'background-color': 'darkgreen',
                           'color': 'white',
                           'border-color': 'darkblack'})
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/aa6519ea-d9ae-4113-9a52-20f1aa6c70ff)

### Step-3: Optimizing the dataset
Processing the data before tarining by removing the unnecessary columns.
```python
df_class.info()
```
In pandas, the `info()` method is used to get a concise summary of a DataFrame. This method provides information about the DataFrame, including the data types of each column, the number of non-null values, and memory usage. It's a handy tool for quickly assessing the structure and content of your DataFrame.
### Simple Breakdown of `info()` Method:
- **Index and Datatype of Each Column:** Shows the name of each column along with the data type of its elements (e.g., int64, float64, object).
- **Non-Null Count:** Indicates the number of non-null (non-missing) values in each column.
- **Memory Usage:** Provides an estimate of the memory usage of the DataFrame.
This method is especially useful when you want to check for missing values, understand the data types in your DataFrame, and get an overall sense of its size and composition. It's often used as a first step in exploring and understanding the characteristics of a dataset.
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/d2492467-b62b-4877-ac1a-71c2414b5d97)

### Removing unnecessary columns
```python
df_class = df_class.drop(['Timestamp','Email ID','Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'],axis=1)
```
### Specifying column names
```python
df_class.columns = ["Name","Branch","Semester","Resourse Person","Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
df_class.sample(5)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/eb07c99c-252f-4cd9-ba72-7f461e48b544)

### Checking for null values and knowing the dimensions
```python
df_class.isnull().sum().sum()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/1b18a228-ddcf-447a-94dc-38d07a1f4249)

```python
# dimension
df_class.shape
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/791a54b9-6cbf-4bc3-99a1-c73302d41e48)


## Step-4: Exploratory Data Analysis
### Creating an rp analysis in percentage
```python
round(df_class["Resourse Person"].value_counts(normalize=True)*100,2)
```
Explanation:
-df_class["Resourse Person"]: This part extracts the column named "Resourse Person" from the DataFrame df_class.
-.value_counts(): This function counts the occurrences of each unique value in the specified column, which is "Resourse Person" in this case.
-normalize=True: The normalize parameter is set to True, which means the counts will be normalized to represent relative frequencies (percentages) instead of absolute counts.
-*100: After normalization, the counts are multiplied by 100 to convert the relative frequencies into percentages.
-round(..., 2): The resulting percentages are then rounded to two decimal places.
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/9c2e99eb-09b7-4030-a27a-6011e49c7b50)

### Creating a percentage analysis of Name-wise distribution of data
```python
round(df_class["Name"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/d017485f-e466-4fb9-911d-0a9864a8fa9d)

### Step-5: Visualization
In this part,we are visualizing the analysed data part using graphs , pie charts etc.
```python
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Resourse Person', data=df_class)
#ax.bar_label(ax.containers[0])
plt.title("Faculty-wise distribution of data", fontsize=20,color = 'Brown',pad=20)
ax =plt.subplot(1,2,2)
ax=df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "Resourse Person", fontsize = 20,color='Brown',pad=20);
```
### Create subplot with 1 row and 2 columns, selecting the first subplot
``ax = plt.subplot(1, 2, 1)``

### Create a count plot using Seaborn
``ax = sns.countplot(x='Resourse Person', data=df_class)``

### Set title for the first subplot
``plt.title("Faculty-wise distribution of data", fontsize=20, color='Brown', pad=20)``

### Move to the second subplot
``ax = plt.subplot(1, 2, 2)``

### Create a pie chart for the distribution of 'Resourse Person'
``ax = df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True)``

### Set title for the pie chart
``ax.set_title(label="Resourse Person", fontsize=20, color='Brown', pad=20)``
This code utilizes Matplotlib and Seaborn to generate a side-by-side visualization of the distribution of a categorical variable ("Resourse Person") in the DataFrame df_class. The first subplot displays a count plot (bar chart), while the second subplot presents a pie chart. Both charts provide insights into the frequency and proportion of different categories in the "Resourse Person" column.
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/7618735c-3670-402b-8f5f-9c116ce22f60)

## Step-6: Creating a summary of responses
 A box and whisker plot or diagram (otherwise known as a boxplot), is a graph summarising a set of data. The shape of the boxplot shows how the data is distributed and it also shows any outliers. It is a useful way to compare different sets of data as you can draw more than one boxplot per graph.
 In this step we are creating box plot on various attributes and resource persons.
 
### 1) Creating boxplot on Content quality v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/0811d02d-09c4-4f0c-a04d-ba90c6a8214a)

### 2) Creating boxplot on Effectiveness v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Effectiveness'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/763665be-4573-4318-8760-9c3c235446a7)

### 3) Creating boxplot on Relevance v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Relevance'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/bb772266-04a8-4741-b1ac-dd5bfe4bcb8a)

### 4) Creating boxplot on Overall Organization v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Overall Organization'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/f0502dab-5a6b-4152-b353-a7d64ebd2e19)

### 5) Creating boxplot on Content quality v/s Branch
```python
sns.boxplot(y=df_class['Branch'],x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/386411f4-a49f-4ca0-9d28-4f136cb4686a)


## Step-7: Unsupervised machine learning
Using K-means Clustering to identify segmentation over student's satisfaction.
### Finding the best value of k using elbow method
# Elbow Method in Machine Learning
The elbow method is a technique used to determine the optimal number of clusters (k) in a clustering algorithm, such as k-means. It involves plotting the sum of squared distances (inertia) against different values of k and identifying the "elbow" point.
### Steps:
1. **Choose a Range of k Values:**
   - Select a range of potential values for the number of clusters.
2. **Run the Clustering Algorithm:**
   - Apply the clustering algorithm (e.g., k-means) for each value of k.
   - Calculate the sum of squared distances (inertia) for each clustering configuration.
3. **Plot the Elbow Curve:**
   - Plot the values of k against the corresponding sum of squared distances.
   - Look for an "elbow" point where the rate of decrease in inertia slows down.
4. **Identify the Elbow:**
   - The optimal k is often at the point where the inertia starts decreasing more slowly, forming an elbow.
### Interpretation:
- The elbow represents a trade-off between minimizing inertia and avoiding overfitting.
- It helps to find a balanced number of clusters for the given dataset.
Remember, while the elbow method is a useful heuristic, other factors like domain knowledge and analysis goals should also be considered in determining the final number of clusters.
```python
input_col=["Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
X=df_class[input_col].values
```
### Initialize an empty list to store the within-cluster sum of squares
```from sklearn.cluster import KMeans
wcss = []
```
### Try different values of k
```python
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# here inertia calculate sum of square distance in each cluster
```
### Plotting sws v/s k value graphs
```python
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/03fdcc07-ccb5-43e9-b90b-0542aa578087)

##  Gridsearch method
Another method which can be used to find the optimized value of k is gridsearch method
```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```
```python
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/4ae0d5e1-8ba8-4f0b-993c-f010f4c81ed6)

## Step-8: Implementing K-means Clustering
K-means Clustering is a model used in unsupervised learning.Here mean values are taken into account after fixing a centroid and the process is repeated.
```python
Perform k-means clusteringprint("Best Parameters:", best_params)
print("Best Score:", best_score)
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)#
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/5b32fec0-b426-420d-8aba-be781bf9630c)

## Extracting labels and cluster centers
Get the cluster labels and centroids
```python
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
df_class.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/83bb23dd-e04d-4a8e-9b4a-3892befaae5a)


### Visualizing the clustering using first two features
```python
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:,1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/733c430d-937f-4dd3-9049-42fc51ee8070)
