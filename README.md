
<H3>ENTER YOUR NAME       : Subash M </H3>
<H3>ENTER YOUR REGISTER NO: 212223040210 </H3>
<H3>EX.NO:1</H3>
<H3>DATE : 28.01.2026 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))


```
## OUTPUT:
### Dataset:

<img width="1590" height="450" alt="image" src="https://github.com/user-attachments/assets/968cb669-4ed3-4743-8c9f-09c5210ca80c" />

### X Values:

<img width="473" height="295" alt="image" src="https://github.com/user-attachments/assets/9bac8886-17d9-409f-918e-aebf3f2f8be3" />


### Y Values:

<img width="302" height="141" alt="image" src="https://github.com/user-attachments/assets/19a13947-9fe9-4bd2-b3a6-13e33824ee9e" />


### Null Values:

<img width="231" height="382" alt="image" src="https://github.com/user-attachments/assets/55d40e0a-cf93-4767-bac1-27e8de0068d2" />


### Duplicated Values:

<img width="267" height="312" alt="image" src="https://github.com/user-attachments/assets/e7e26a57-e7e6-4cd9-8a48-0b0d8660b055" />


### Description:

<img width="1557" height="310" alt="image" src="https://github.com/user-attachments/assets/e0add98b-faec-4466-a8c0-60065091de11" />


### Normalized Dataset:

<img width="827" height="666" alt="image" src="https://github.com/user-attachments/assets/0eecdbfb-8330-49e6-9859-1734af70674f" />


### Training Data:

<img width="725" height="213" alt="image" src="https://github.com/user-attachments/assets/4116d0f1-d750-4145-8f4a-17da436eceaa" />


### Testing Data:


<img width="726" height="202" alt="image" src="https://github.com/user-attachments/assets/e4e4a7f9-6f09-4877-bbfc-5e64c0f08332" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
