# Pandas Library in Python for Machine Learning: Simplifying Data Handling

## Introduction
Pandas is a powerful Python library widely used in machine learning for data manipulation and analysis. It offers intuitive data structures and functions that streamline the process of preparing data for machine learning models. This short note aims to highlight the essential role of Pandas in the machine learning workflow.

## Key Features
1. **Data Loading and Exploration**:
   - Pandas simplifies the loading of data from various sources such as CSV, Excel, SQL databases, and more into a DataFrame, a two-dimensional labeled data structure.
   - It provides convenient methods for data exploration, allowing users to quickly understand the structure and characteristics of the dataset.

2. **Data Preprocessing**:
   - Handling Missing Data: Pandas offers robust mechanisms to handle missing data, including methods for detection, removal, and imputation, ensuring the integrity of the dataset.
   - Data Transformation: Pandas facilitates data transformation tasks such as normalization, standardization, and encoding categorical variables, crucial steps in preprocessing data for machine learning models.

3. **Feature Engineering**:
   - Creating New Features: Pandas enables the creation of new features through combinations of existing ones or by applying custom functions, empowering users to extract valuable insights from the data.
   - Dimensionality Reduction: Techniques like Principal Component Analysis (PCA) can be applied to Pandas DataFrames to reduce dimensionality and improve model performance.

4. **Data Splitting and Sampling**:
   - Pandas provides functionalities for splitting datasets into training, validation, and test sets, essential for evaluating model performance.
   - Sampling methods such as random sampling or stratified sampling can be easily implemented using Pandas, allowing for effective data partitioning.

5. **Integration with Machine Learning Libraries**:
   - Pandas seamlessly integrates with popular machine learning libraries such as Scikit-learn, TensorFlow, and PyTorch, facilitating data preprocessing and model training workflows.
   - DataFrames can be directly used as inputs to machine learning algorithms, enabling smooth transitions between data manipulation and model development stages.

# Dealing with Outliers

In statistics, an outlier is a data point that differs significantly from other observations.An outlier may be due to variability in the measurement or it may indicate experimental error; the latter are sometimes excluded from the data set. An outlier can cause serious problems in statistical analyses.
Remember that even if a data point is an outlier, its still a data point! Carefully consider your data, its sources, and your goals whenver deciding to remove an outlier. Each case is different!

## Lecture Goals
* Understand different mathmatical definitions of outliers
* Use Python tools to recognize outliers and remove them

## Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
## Generating Data
```python
# Choose a mean,standard deviation, and number of samples
def create_ages(mu=50,sigma=13,num_samples=100,seed=42):
    # Set a random seed in the same cell as the random call to get the same values as us
    # We set seed to 42 (42 is an arbitrary choice from Hitchhiker's Guide to the Galaxy)
    np.random.seed(seed)
    sample_ages = np.random.normal(loc=mu,scale=sigma,size=num_samples)
    sample_ages = np.round(sample_ages,decimals=0)
    return sample_ages
sample = create_ages()
sample
```
    array([56., 48., 58., 70., 47., 47., 71., 60., 44., 57., 44., 44., 53.,
           25., 28., 43., 37., 54., 38., 32., 69., 47., 51., 31., 43., 51.,
           35., 55., 42., 46., 42., 74., 50., 36., 61., 34., 53., 25., 33.,
           53., 60., 52., 48., 46., 31., 41., 44., 64., 54., 27., 54., 45.,
           41., 58., 63., 62., 39., 46., 54., 63., 44., 48., 36., 34., 61.,
           68., 49., 63., 55., 42., 55., 70., 50., 70., 16., 61., 51., 46.,
           51., 24., 47., 55., 69., 43., 39., 43., 62., 54., 43., 57., 51.,
           63., 41., 46., 45., 31., 54., 53., 50., 47.])

## Visualize and Describe the Data
```python
sns.distplot(sample,bins=10,kde=False)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/a3dfc6cb-317d-478c-9387-24b0f0625cb0)

```python
sns.boxplot(sample)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/6b0893d0-39d4-4162-8ccb-ee8101d17967)


```python
ser = pd.Series(sample)
ser.describe()
```
    count    100.00000
    mean      48.66000
    std       11.82039
    min       16.00000
    25%       42.00000
    50%       48.00000
    75%       55.25000
    max       74.00000
    dtype: float64

## Trimming or Fixing Based Off Domain Knowledge
If we know we're dealing with a dataset pertaining to voting age (18 years old in the USA), then it makes sense to either drop anything less than that OR fix values lower than 18 and push them up to 18.
```python
ser[ser > 18]
```
    0     56.0
    1     48.0
    2     58.0
    3     70.0
    4     47.0
          ... 
    95    31.0
    96    54.0
    97    53.0
    98    50.0
    99    47.0
    Length: 99, dtype: float64

```python
# It dropped one person
len(ser[ser > 18])
```
    99
```python
def fix_values(age):
    
    if age < 18:
        return 18
    else:
        return age
# "Fixes" one person's age
ser.apply(fix_values)
```
    0     56.0
    1     48.0
    2     58.0
    3     70.0
    4     47.0
          ... 
    95    31.0
    96    54.0
    97    53.0
    98    50.0
    99    47.0
    Length: 100, dtype: float64

```python
len(ser.apply(fix_values))
```
    100

There are many ways to identify and remove outliers:
* Trimming based off a provided value
* Capping based off IQR or STD
* https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba
* https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623

## Ames Data Set
Let's explore any extreme outliers in our Ames Housing Data Set

```python
df = pd.read_csv("../DATA/Ames_Housing_Data.csv")
df.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/76030300-bfdf-4b35-95c0-1c91dd11a3c9)

```python
sns.heatmap(df.corr())
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/deb10e3e-0712-4676-9048-ae838bb37958)

```python
df.corr()['SalePrice'].sort_values()
```
    PID               -0.246521
    Enclosed Porch    -0.128787
    Kitchen AbvGr     -0.119814
    Overall Cond      -0.101697
    MS SubClass       -0.085092
    Low Qual Fin SF   -0.037660
    Bsmt Half Bath    -0.035835
    Yr Sold           -0.030569
    Misc Val          -0.015691
    BsmtFin SF 2       0.005891
    3Ssn Porch         0.032225
    Mo Sold            0.035259
    Pool Area          0.068403
    Screen Porch       0.112151
    Bedroom AbvGr      0.143913
    Bsmt Unf SF        0.182855
    Lot Area           0.266549
    2nd Flr SF         0.269373
    Bsmt Full Bath     0.276050
    Half Bath          0.285056
    Open Porch SF      0.312951
    Wood Deck SF       0.327143
    Lot Frontage       0.357318
    BsmtFin SF 1       0.432914
    Fireplaces         0.474558
    TotRms AbvGrd      0.495474
    Mas Vnr Area       0.508285
    Garage Yr Blt      0.526965
    Year Remod/Add     0.532974
    Full Bath          0.545604
    Year Built         0.558426
    1st Flr SF         0.621676
    Total Bsmt SF      0.632280
    Garage Area        0.640401
    Garage Cars        0.647877
    Gr Liv Area        0.706780
    Overall Qual       0.799262
    SalePrice          1.000000
    Name: SalePrice, dtype: float64

```python
sns.distplot(df["SalePrice"])
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/89300dc4-5c6a-4f78-a389-9f79e2e2a329)

```python
sns.scatterplot(x='Overall Qual',y='SalePrice',data=df)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/57c6c531-eba4-4801-b79c-c7594523f58c)

```python
df[(df['Overall Qual']>8) & (df['SalePrice']<200000)]
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/51167c6b-34a0-4cad-ae55-bd5a858b71de)

```python
sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/57248a1a-3cab-442c-b86f-706beef04791)

```python
df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)]
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/8643f9b9-fd21-4632-8981-585c06ae6847)

```python
df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)].index
```
    Int64Index([1498, 2180, 2181], dtype='int64')

```python
ind_drop = df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)].index
df = df.drop(ind_drop,axis=0)
sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/0cde059c-7d1c-4139-9b47-422c876cc440)

```python
sns.scatterplot(x='Overall Qual',y='SalePrice',data=df)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/fa7849e2-a502-4181-a118-7b51d3abe839)

```python
df.to_csv("../DATA/Ames_outliers_removed.csv",index=False)
```

# Series in Pandas for Machine Learning
In machine learning, particularly when dealing with tabular data, the Pandas library in Python is a powerful tool for data manipulation and analysis. Series in Pandas represent one-dimensional labeled arrays and are a fundamental data structure for handling data in Pandas. Here's a short note on using Series in Pandas for machine learning:

## Series in Pandas:
- A Series is a one-dimensional array-like object containing an array of data (of any NumPy data type) and an associated array of data labels, called its index.
- Series can hold any data type: integers, floats, strings, Python objects, etc.
- Series provides powerful indexing capabilities, making it easy to access and manipulate data.
- Series can be thought of as a column in a table or as a single column of a DataFrame, which is a two-dimensional tabular data structure in Pandas.

## Using Series in Machine Learning:
1. **Data Preparation:**
   - Series can be used to represent features (independent variables) or target variables (dependent variables) in machine learning models.
   - Series can be created from lists, arrays, dictionaries, or even from other Series, providing flexibility in data preparation.
2. **Data Exploration:**
   - Series provides various methods and attributes for data exploration, such as descriptive statistics (mean, median, standard deviation, etc.), unique values, value counts, etc.
   - This allows machine learning practitioners to gain insights into the distribution and characteristics of the data before model training.
3. **Data Preprocessing:**
   - Series can be used for data preprocessing tasks like handling missing values, scaling, normalization, encoding categorical variables, etc.
   - Pandas provides convenient methods for handling missing data, such as `dropna()`, `fillna()`, and `interpolate()`.
4. **Integration with Machine Learning Models:**
   - Series can be directly used as input features or target variables for training machine learning models.
   - Pandas Series can be seamlessly integrated with popular machine learning libraries like scikit-learn, enabling smooth workflow in model training and evaluation.
5. **Feature Engineering:**
   - Series can be manipulated to create new features through operations like arithmetic operations, string manipulation, datetime operations, etc.
   - Feature engineering using Series can help improve model performance by providing more meaningful and relevant information to the models.

## Imports
```python
import numpy as np
import pandas as pd
```
    /home/u213213/tmp/ipykernel_4065176/1662815981.py:2: DeprecationWarning: 
    Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),
    (to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)
    but was not found to be installed on your system.
    If this would cause problems for you,
    please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466
            
      import pandas as pd
      
### Index and Data Lists
```python
myindex = ['USA','Canada','Mexico']
mydata = [1776,1867,1821]
myser = pd.Series(data=mydata)
myser
```
    0    1776
    1    1867
    2    1821
    dtype: int64

```python
pd.Series(data=mydata,index=myindex)
```
    USA       1776
    Canada    1867
    Mexico    1821
    dtype: int64

```python
ran_data = np.random.randint(0,100,4)
ran_data
```
    array([78, 42, 14,  8])

```python
names = ['Andrew','Bobo','Claire','David']
ages = pd.Series(ran_data,names)
ages
```
    Andrew    78
    Bobo      42
    Claire    14
    David      8
    dtype: int64

### From a  Dictionary
```python
ages = {'Sammy':5,'Frank':10,'Spike':7}
ages
```
    {'Sammy': 5, 'Frank': 10, 'Spike': 7}

```python
pd.Series(ages)
```
    Sammy     5
    Frank    10
    Spike     7
    dtype: int64

# Key Ideas of a Series
## Named Index
```python
# Imaginary Sales Data for 1st and 2nd Quarters for Global Company
q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}
q2 = {'Brazil': 100,'China': 500, 'India': 210,'USA': 260}
# Convert into Pandas Series
sales_Q1 = pd.Series(q1)
sales_Q2 = pd.Series(q2)
sales_Q1
```
    Japan     80
    China    450
    India    200
    USA      250
    dtype: int64

```python
# Call values based on Named Index
sales_Q1['Japan']
```
    80

## Operations
```python
# Grab just the index keys
sales_Q1.keys()
```
    Index(['Japan', 'China', 'India', 'USA'], dtype='object')

```python
# Can Perform Operations Broadcasted across entire Series
sales_Q1 * 2
```
    Japan    160
    China    900
    India    400
    USA      500
    dtype: int64

```python
sales_Q2 / 100
```
    Brazil    1.0
    China     5.0
    India     2.1
    USA       2.6
    dtype: float64

## Between Series
```python
# Notice how Pandas informs you of mismatch with NaN
sales_Q1 + sales_Q2
```
    /home/u213213/.local/lib/python3.9/site-packages/pandas/io/formats/format.py:1458: RuntimeWarning: invalid value encountered in greater
      has_large_values = (abs_vals > 1e6).any()
    /home/u213213/.local/lib/python3.9/site-packages/pandas/io/formats/format.py:1459: RuntimeWarning: invalid value encountered in less
      has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
    /home/u213213/.local/lib/python3.9/site-packages/pandas/io/formats/format.py:1459: RuntimeWarning: invalid value encountered in greater
      has_small_values = ((abs_vals < 10 ** (-self.digits)) & (abs_vals > 0)).any()
    Brazil      NaN
    China     950.0
    India     410.0
    Japan       NaN
    USA       510.0
    dtype: float64

```python
# You can fill these with any value you want
sales_Q1.add(sales_Q2,fill_value=0)
```
    Brazil    100.0
    China     950.0
    India     410.0
    Japan      80.0
    USA       510.0
    dtype: float64

That is all we need to know about Series, up next, DataFrames!

# DataFrames in Pandas for Machine Learning
DataFrames in Pandas are essential for machine learning tasks due to their tabular structure and versatile functionalities:
**1. Tabular Representation:**
- DataFrames organize data into rows and columns, resembling spreadsheet tables, facilitating easy data manipulation and exploration.
**2. Input Handling:**
- They support diverse data formats like CSV, Excel, SQL, and JSON, simplifying data loading and integration into machine learning pipelines.
**3. Data Exploration and Preparation:**
- DataFrames provide methods for summarizing data, handling missing values, and preparing data for training, enabling efficient data preprocessing.
**4. Feature Engineering:**
- With Pandas, users can create new features by manipulating existing ones, essential for enhancing model performance.
**5. Integration with ML Libraries:**
- DataFrames seamlessly integrate with popular machine learning libraries like scikit-learn, TensorFlow, and PyTorch, facilitating smooth data preprocessing and model training.
**6. Post-Modeling Analysis:**
- They aid in analyzing model predictions, evaluating performance metrics, and generating diagnostic plots, essential for assessing model effectiveness.

# DataFrames

Throughout the course, most of our data exploration will be done with DataFrames. DataFrames are an extremely powerful tool and a natural extension of the Pandas Series. By definition all a DataFrame is:

**A Pandas DataFrame consists of multiple Pandas Series that share index values.**

## Imports
```python
import numpy as np
import pandas as pd
```

### Creating a DataFrame from Python Objects
```python
# help(pd.DataFrame)
# Make sure the seed is in the same cell as the random call
# https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
np.random.seed(101)
mydata = np.random.randint(0,101,(4,3))
mydata
```
    array([[95, 11, 81],
           [70, 63, 87],
           [75,  9, 77],
           [40,  4, 63]])

```python
myindex = ['CA','NY','AZ','TX']
mycolumns = ['Jan','Feb','Mar']
df = pd.DataFrame(data=mydata)
df
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/c4f3dcde-15a8-49e7-930f-ac1d7f782366)

```python
df = pd.DataFrame(data=mydata,index=myindex)
df
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/d2f4500c-e856-4598-b13e-e318f2221b4f)

```python
df = pd.DataFrame(data=mydata,index=myindex,columns=mycolumns)
df 
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/dbf561a9-3f53-4633-8fb3-70bfde442e21)

```python
df.info()
```
    <class 'pandas.core.frame.DataFrame'>
    Index: 4 entries, CA to TX
    Data columns (total 3 columns):
    Jan    4 non-null int32
    Feb    4 non-null int32
    Mar    4 non-null int32
    dtypes: int32(3)
    memory usage: 80.0+ bytes

# Reading a .csv file for a DataFrame
## NOTE: We will go over all kinds of data inputs and outputs (.html, .csv, .xlxs , etc...) later on in the course! For now we just need to read in a simple .csv file.
## CSV
Comma Separated Values files are text files that use commas as field delimeters.<br>
Unless you're running the virtual environment included with the course, you may need to install <tt>xlrd</tt> and <tt>openpyxl</tt>.<br>
In your terminal/command prompt run:

    conda install xlrd
    conda install openpyxl
Then restart Jupyter Notebook.
(or use pip install if you aren't using the Anaconda Distribution)

### Understanding File Paths
You have two options when reading a file with pandas:

1. If your .py file or .ipynb notebook is located in the **exact** same folder location as the .csv file you want to read, simply pass in the file name as a string, for example:

        df = pd.read_csv('some_file.csv')
2. Pass in the entire file path if you are located in a different directory. The file path must be 100% correct in order for this to work. For example:

        df = pd.read_csv("C:\\Users\\myself\\files\\some_file.csv")

#### Print your current directory file path with pwd
```python
pwd
```
    'C:\\Users\\Marcial\\Pierian-Data-Courses\\Machine-Learning-MasterClass\\03-Pandas'

#### List the files in your current directory with ls
```python
ls
```
     Volume in drive C has no label.
     Volume Serial Number is 3652-BD2F
    
     Directory of C:\Users\Marcial\Pierian-Data-Courses\Machine-Learning-MasterClass\03-Pandas
    
    06/30/2020  05:21 PM    <DIR>          .
    06/30/2020  05:21 PM    <DIR>          ..
    01/27/2020  01:55 PM    <DIR>          .ipynb_checkpoints
    06/30/2020  04:51 PM           565,390 00-Series.ipynb
    06/30/2020  05:21 PM           207,278 01-DataFrames.ipynb
    01/27/2020  06:24 PM           194,565 02-Conditional-Filtering.ipynb
    06/30/2020  11:41 AM            82,092 03-Useful-Methods.ipynb
    06/30/2020  11:41 AM            45,221 04-Missing-Data.ipynb
    06/30/2020  11:42 AM             1,101 05-Groupby-Operations.ipynb
    06/30/2020  11:42 AM             1,103 06-Combining-DataFrames.ipynb
    06/30/2020  11:42 AM             1,095 07-Text-Methods.ipynb
    06/30/2020  11:42 AM             1,095 08-Time-Methods.ipynb
    06/30/2020  11:42 AM             1,101 09-Inputs-and-Outputs.ipynb
    06/30/2020  11:42 AM             1,095 10-Simple-Plots.ipynb
    06/30/2020  11:42 AM               951 11-Pandas-Project-Exercise.ipynb
    06/30/2020  11:42 AM             1,118 12-Pandas-Project-Exercise-Solution.ipynb
    02/07/2020  12:26 PM               177 movie_scores.csv
    01/27/2020  02:28 PM            18,752 tips.csv
                  15 File(s)      1,122,134 bytes
                   3 Dir(s)  84,920,594,432 bytes free

# DataFrames
## Obtaining Basic Information About DataFrame
```python
df.columns
```
    Index(['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size',
           'price_per_person', 'Payer Name', 'CC Number', 'Payment ID'],
          dtype='object')

```python
df.index
```
    RangeIndex(start=0, stop=244, step=1)

```python
df.head(3)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/05e6122f-590d-42e0-b076-12fdd42f2d8d)


```python
df.tail(3)
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/7d602120-2555-47ad-95b3-f22bb1b35a14)

```python
df.info()
```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 244 entries, 0 to 243
    Data columns (total 11 columns):
    total_bill          244 non-null float64
    tip                 244 non-null float64
    sex                 244 non-null object
    smoker              244 non-null object
    day                 244 non-null object
    time                244 non-null object
    size                244 non-null int64
    price_per_person    244 non-null float64
    Payer Name          244 non-null object
    CC Number           244 non-null int64
    Payment ID          244 non-null object
    dtypes: float64(3), int64(2), object(6)
    memory usage: 21.0+ KB

```python
len(df)
```
    244

```python
df.describe()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/d9344fe5-5f6f-4b01-9998-510a2ccfa74e)


## Selection and Indexing
Let's learn how to retrieve information from a DataFrame.

### COLUMNS
We will begin be learning how to extract information based on the columns
```python
df.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/eebc56b0-8c9c-40d7-8656-dec9776f1f46)

#### Grab a Single Column
```python
df['total_bill']
```
    0      16.99
    1      10.34
    2      21.01
    3      23.68
    4      24.59
           ...  
    239    29.03
    240    27.18
    241    22.67
    242    17.82
    243    18.78
    Name: total_bill, Length: 244, dtype: float64

```python
type(df['total_bill'])
```
    pandas.core.series.Series

#### Grab Multiple Columns
```python
# Note how its a python list of column names! Thus the double brackets.
df[['total_bill','tip']]
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/7a6e316c-ebec-473d-b3f7-c3983ea56b24)


#### Create New Columns
```python
df['tip_percentage'] = 100* df['tip'] / df['total_bill']
df.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/3e756df3-f3eb-4325-90f8-8542005d1b62)


#### Adjust Existing Columns
```python
# Because pandas is based on numpy, we get awesome capabilities with numpy's universal functions!
df['price_per_person'] = np.round(df['price_per_person'],2)
df.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/05d1f691-d258-431f-8ab1-4c49b296e9f4)


#### Remove Columns
```python
# df.drop('tip_percentage',axis=1)
df = df.drop("tip_percentage",axis=1)
df.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/dfddb860-02c5-4a54-a6ce-cb76426c3c5f)

### ROWS
```python
df.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/2aef413b-2d28-4b49-829e-8bf79655dc4f)

```python
df = df.set_index('Payment ID')
df.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/dbf56835-aa59-43e5-bafd-029a14f887b5)


#### Grab a Single Row
```python
# Integer Based
df.iloc[0]
```
    total_bill                       16.99
    tip                               1.01
    sex                             Female
    smoker                              No
    day                                Sun
    time                            Dinner
    size                                 2
    price_per_person                  8.49
    Payer Name          Christy Cunningham
    CC Number             3560325168603410
    Name: Sun2959, dtype: object

```python
# Name Based
df.loc['Sun2959']
```
    total_bill                       16.99
    tip                               1.01
    sex                             Female
    smoker                              No
    day                                Sun
    time                            Dinner
    size                                 2
    price_per_person                  8.49
    Payer Name          Christy Cunningham
    CC Number             3560325168603410
    Name: Sun2959, dtype: object

#### Grab Multiple Rows
```python
df.iloc[0:4]
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/cd791e75-068f-4c89-8633-64a01dfe57e8)


#### Remove Row
```python
df.drop('Sun2959',axis=0).head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/1fab7fbc-7e7c-487d-9145-f360b7aeaa3e)

# Dealing with Missing Data
We already reviewed Pandas operations for missing data, now let's apply this to clean a real data file. Keep in mind, there is no 100% correct way of doing this, and this notebook just serves as an example of some reasonable approaches to take on this data.

#### Note: Throughout this section we will be slowly cleaning and adding features to the Ames Housing Dataset for use in the next section. Make sure to always be loading the same file name as in the notebook.
#### 2nd Note: Some of the methods shown here may not lead to optimal performance, but instead are shown to display examples of various methods available.

## Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
    /home/u213213/tmp/ipykernel_4134295/265554930.py:2: DeprecationWarning: 
    Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),
    (to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)
    but was not found to be installed on your system.
    If this would cause problems for you,
    please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466    
      import pandas as pd

## Data
```python
df = pd.read_c
df.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/4b30389c-f300-4369-b38a-be6e3e787c37)

```python
len(df.columns)
```
    81
    
```python
df.info()
```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2927 entries, 0 to 2926
    Data columns (total 81 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   PID              2927 non-null   int64  
     1   MS SubClass      2927 non-null   int64  
     2   MS Zoning        2927 non-null   object 
     3   Lot Frontage     2437 non-null   float64
     4   Lot Area         2927 non-null   int64  
     5   Street           2927 non-null   object 
     6   Alley            198 non-null    object 
     7   Lot Shape        2927 non-null   object 
     8   Land Contour     2927 non-null   object 
     9   Utilities        2927 non-null   object 
     10  Lot Config       2927 non-null   object 
     11  Land Slope       2927 non-null   object 
     12  Neighborhood     2927 non-null   object 
     13  Condition 1      2927 non-null   object 
     14  Condition 2      2927 non-null   object 
     15  Bldg Type        2927 non-null   object 
     16  House Style      2927 non-null   object 
     17  Overall Qual     2927 non-null   int64  
     18  Overall Cond     2927 non-null   int64  
     19  Year Built       2927 non-null   int64  
     20  Year Remod/Add   2927 non-null   int64  
     21  Roof Style       2927 non-null   object 
     22  Roof Matl        2927 non-null   object 
     23  Exterior 1st     2927 non-null   object 
     24  Exterior 2nd     2927 non-null   object 
     25  Mas Vnr Type     1152 non-null   object 
     26  Mas Vnr Area     2904 non-null   float64
     27  Exter Qual       2927 non-null   object 
     28  Exter Cond       2927 non-null   object 
     29  Foundation       2927 non-null   object 
     30  Bsmt Qual        2847 non-null   object 
     31  Bsmt Cond        2847 non-null   object 
     32  Bsmt Exposure    2844 non-null   object 
     33  BsmtFin Type 1   2847 non-null   object 
     34  BsmtFin SF 1     2926 non-null   float64
     35  BsmtFin Type 2   2846 non-null   object 
     36  BsmtFin SF 2     2926 non-null   float64
     37  Bsmt Unf SF      2926 non-null   float64
     38  Total Bsmt SF    2926 non-null   float64
     39  Heating          2927 non-null   object 
     40  Heating QC       2927 non-null   object 
     41  Central Air      2927 non-null   object 
     42  Electrical       2926 non-null   object 
     43  1st Flr SF       2927 non-null   int64  
     44  2nd Flr SF       2927 non-null   int64  
     45  Low Qual Fin SF  2927 non-null   int64  
     46  Gr Liv Area      2927 non-null   int64  
     47  Bsmt Full Bath   2925 non-null   float64
     48  Bsmt Half Bath   2925 non-null   float64
     49  Full Bath        2927 non-null   int64  
     50  Half Bath        2927 non-null   int64  
     51  Bedroom AbvGr    2927 non-null   int64  
     52  Kitchen AbvGr    2927 non-null   int64  
     53  Kitchen Qual     2927 non-null   object 
     54  TotRms AbvGrd    2927 non-null   int64  
     55  Functional       2927 non-null   object 
     56  Fireplaces       2927 non-null   int64  
     57  Fireplace Qu     1505 non-null   object 
     58  Garage Type      2770 non-null   object 
     59  Garage Yr Blt    2768 non-null   float64
     60  Garage Finish    2768 non-null   object 
     61  Garage Cars      2926 non-null   float64
     62  Garage Area      2926 non-null   float64
     63  Garage Qual      2768 non-null   object 
     64  Garage Cond      2768 non-null   object 
     65  Paved Drive      2927 non-null   object 
     66  Wood Deck SF     2927 non-null   int64  
     67  Open Porch SF    2927 non-null   int64  
     68  Enclosed Porch   2927 non-null   int64  
     69  3Ssn Porch       2927 non-null   int64  
     70  Screen Porch     2927 non-null   int64  
     71  Pool Area        2927 non-null   int64  
     72  Pool QC          12 non-null     object 
     73  Fence            572 non-null    object 
     74  Misc Feature     105 non-null    object 
     75  Misc Val         2927 non-null   int64  
     76  Mo Sold          2927 non-null   int64  
     77  Yr Sold          2927 non-null   int64  
     78  Sale Type        2927 non-null   object 
     79  Sale Condition   2927 non-null   object 
     80  SalePrice        2927 non-null   int64  
    dtypes: float64(11), int64(27), object(43)
    memory usage: 1.8+ MB

### Mas Vnr Feature 
Based on the Description Text File, Mas Vnr Type and Mas Vnr Area being missing (NaN) is likely to mean the house simply just doesn't have a masonry veneer, in which case, we will fill in this data as we did before.
```python
df["Mas Vnr Type"] = df["Mas Vnr Type"].fillna("None")
df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)
percent_nan = percent_missing(df)
sns.barplot(x=percent_nan.index,y=percent_nan)
plt.xticks(rotation=90);
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/05c09ba6-2014-442c-aff7-c23d8cc39c94)


# Conditional Filtering

## Imports
```python
import numpy as np
import pandas as pd
```
    /home/u213213/tmp/ipykernel_4063799/1662815981.py:2: DeprecationWarning: 
    Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),
    (to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)
    but was not found to be installed on your system.
    If this would cause problems for you,
    please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466
            
      import pandas as pd

```python
df = pd.read_csv('tips.csv')
df.head()
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/cd4ceddc-954d-4f26-a53f-85c24c036fa5)


## Conditions
```python
# df['total_bill'] > 30
bool_series = df['total_bill'] > 30
df[bool_series]
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/18078214-897f-4757-94ff-9b98c461dea4)

```python
df[df['sex'] == 'Male']
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/a6c24d36-b8f2-4b4c-ac16-6cdeaeb32abf)

## Multiple Conditions
Recall the steps:
* Get the conditions
* Wrap each condition in parenthesis
* Use the | or & operator, depending if you want an 
    * OR | (either condition is True)
    * AND & (both conditions must be True)
* You can also use the ~ operator as a NOT operation

```python
df[(df['total_bill'] > 30) & ~(df['sex']=='Male')]
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/537bc7b5-5b14-4169-b9a0-94bdd026f73e)

```python
df[(df['total_bill'] > 30) & (df['sex']!='Male')]
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/d2229c8f-8416-4c70-b557-e31a2630ffa4)

```python
# The Weekend
df[(df['day'] =='Sun') | (df['day']=='Sat')]
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/c4413408-3ed4-4d12-89ae-1d78a5bd5f62)


## Conditional Operator isin()
We can use .isin() operator to filter by a list of options.
```python
options = ['Sat','Sun']
df['day'].isin(options)
```
    0       True
    1       True
    2       True
    3       True
    4       True
           ...  
    239     True
    240     True
    241     True
    242     True
    243    False
    Name: day, Length: 244, dtype: bool

```python
df[df['day'].isin(['Sat','Sun'])]
```
![image](https://github.com/Shalin-01/Python-for-ML/assets/145157389/4fcf7c7d-7c15-4491-b5a7-ab53b4d371e2)
