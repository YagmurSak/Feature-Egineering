import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib.pyplot import ylabel, xlabel
from mpmath import zeros

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv(r"C:\Users\Yagmu\OneDrive\Masaüstü\DATA SCIENCE BOOTCAMP\5- Feature Engineering\diabetes-project\diabetes.csv")
df.head()

# Data Analysis and First Insights

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Determinig numerical and categorical variables

# cat_cols = categorical columns
# num_but_cat = seems numerical but categorical
# cat_but_car = seems categorical but cardinal
# num_cols = numerical columns

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

# The mean of the target variable according to categorical variables


def num_cols_analysis(dataframe, column, target):
    outcome_mean = dataframe.groupby("Outcome")[column].mean()
    print(outcome_mean)

num_cols_analysis(df,num_cols,"Outcome")

# Analysis of Outliers

sns.boxplot(x=df["Age"])
plt.show()

plt.figure(figsize=(12, 8))
for i, column in enumerate(num_cols, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(y=df[column], width=0.5)
    plt.axhline(y=df[column].median(), color='red', linestyle='--', linewidth=2)
    plt.title(f'Boxplot of {column}', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

plt.show()

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low , up = outlier_thresholds(df, num_cols)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, num_cols)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)
    

# Missing Value Analysis

df.isnull().values.any()

df.isnull().sum().sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

# Correlation Analysis

corr = df.corr()
corr_df= corr.unstack().sort_values(ascending=False)

corr_df= pd.DataFrame(corr_df)
corr_df.reset_index(inplace=True)
corr_df.columns=["var1","var2","corr"]

corr_df[corr_df["corr"].apply(lambda x: x!=1)].head(10)

#heatmap
sns.set(rc={'figure.figsize': (10, 10)})
sns.heatmap(corr, cmap="RdBu")
plt.show()


### Feature Engineering ###

# There are no missing observations in the data set, but observation units containing the value 0 in variables such as
# Glucose, Insulin, etc. may represent the missing value. Considering this situation, we will assign zero values
# as NaN in the relevant values and then apply the operations to the missing values.

zeros_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def replace_zeros(dataframe, column):
    dataframe[column] = dataframe[column].replace(0, np.nan)
    return dataframe


replace_zeros(df, zeros_columns)

df.isnull().sum()

for col in zeros_columns:
    df[col] = df[col].fillna(df.groupby("Outcome")[col].transform("mean"))

df.head()


# Creating New Variables

df["Age"].min()
df["Age"].max()

df["AGE_NEW"]= pd.cut(df["Age"],bins= [20,45,max(df["Age"])], labels=["mature","senior"])


df["Glucose"].min()
df["Glucose"].max()
df["Glucose"].mean()

df["GLUCOSE_NEW"]= pd.cut(df["Glucose"], bins=[36, 100, 140 , max(df["Glucose"])], labels=["low","normal","high"])

df["BMI_NEW"]=pd.cut(df["BMI"], bins=[18,25,32,max(df["BMI"])], labels=["Normal Weight","Overweight","Obese"])

df.loc[df["Insulin"]<=120,"INSULIN_NEW"]="normal"
df.loc[df["Insulin"]>120, "INSULIN_NEW"]="anormal"

cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols=[ col for col in cat_cols if col != "Outcome"]

df.head()

#Label Encoder

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in cat_cols if df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df,col)

df.head()

#One Hot Encoder

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols = [col for col in cat_cols if col not in binary_cols]

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

selected_columns = ["GLUCOSE_NEW_normal", "GLUCOSE_NEW_high", "BMI_NEW_Overweight", "BMI_NEW_Obese"]

for col in selected_columns:
    df[col] = df[col].apply(lambda x: 1 if x == "True" else 0)

df.head()

# Standardization for numerical variables

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

df[num_cols].head()

# Modelling

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

rf_model = RandomForestClassifier(random_state=46)
rf_model.fit(X_train, y_train)
y_pred_1 = rf_model.predict(X_test)
rf_accuracy= accuracy_score(y_pred_1, y_test)
rf_accuracy

