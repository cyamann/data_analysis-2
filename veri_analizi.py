#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('C:/ai4i2020.csv')
df.head(5)


# In[3]:


# Bar plot: Machine failure vs. Type
plt.figure(figsize=(8, 6))
sns.barplot(x='Machine failure', y='Torque [Nm]', data=df)
plt.title('Machine failure vs. Type')
plt.show()


# In[4]:


# Box plot: Machine failure vs. Air temperature [K]
plt.figure(figsize=(8, 6))
sns.boxplot(x='Machine failure', y='Rotational speed [rpm]', data=df)
plt.title('Machine failure vs. Air temperature [K]')
plt.show()


# In[5]:


# Bar plot: Machine failure vs. Process temperature [K]
plt.figure(figsize=(8, 6))
sns.boxplot(x='Machine failure', y='Tool wear [min]', data=df)
plt.title('Machine failure vs. Process temperature [K]')
plt.show()


# In[7]:


# Bar plot: Machine failure vs. Type
plt.figure(figsize=(8, 6))
sns.boxplot(x='Machine failure', y='Type', data=df)
plt.title('Machine failure vs. Type')
plt.show()


# In[11]:


sns.boxplot(x = df["Air temperature [K]"])


# In[12]:


sns.boxplot(x = df["Process temperature [K]"])


# In[13]:


sns.boxplot(x = df["Rotational speed [rpm]"])


# In[18]:


q1 = df["Rotational speed [rpm]"].quantile(0.25)
q3 = df["Rotational speed [rpm]"].quantile(0.75)
iqr = q3-q1
upper_limit = q3 + 1.5* iqr
anormality = df["Rotational speed [rpm]"] > upper_limit
df.loc[anormality,  "Rotational speed [rpm]"] = upper_limit
sns.boxplot(x = df["Rotational speed [rpm]"]);


# In[22]:


df_torque = df["Torque [Nm]"]
sns.boxplot(x = df_torque);


# In[23]:


q1 = df["Torque [Nm]"].quantile(0.25)
q3 = df["Torque [Nm]"].quantile(0.75)
iqr = q3-q1
upper_limit = q3 + iqr * 1.5
lower_limit = q1 - iqr * 1.5
lower_anormality = df["Torque [Nm]"] < lower_limit
upper_anormality = df["Torque [Nm]"] > upper_limit
df.loc[upper_anormality , "Torque [Nm]"] = upper_limit
df.loc[lower_anormality , "Torque [Nm]"] = lower_limit
sns.boxplot(x = df["Torque [Nm]"]);




# In[24]:


sns.boxplot(x = df["Tool wear [min]"]);


# In[25]:


q1 = df["Tool wear [min]"].quantile(0.25)
q3 = df["Tool wear [min]"].quantile(0.75)
iqr = q3-q1
upper_limit = q3 + iqr * 1.5
anormality = df["Tool wear [min]"] > upper_limit
df.loc[anormality , "Tool wear [min]"] = upper_limit
sns.boxplot(x = df["Tool wear [min]"]);


# In[26]:


df.head()


# In[28]:


from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
df["Encoded Type"] = lbe.fit_transform(df["Type"])


# In[29]:


### Dummy Method / One Hot Method
df_one_hot = pd.get_dummies(df , columns = ["Type"] , prefix = ["Type"])
df[df_one_hot.columns] = df_one_hot


# In[30]:


df.head()


# In[35]:


from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
df["Type"] = df["Type"].astype(str)



# In[36]:


df.head()


# In[37]:


data = {
    "Type": ["M","L","H"]
}

df = pd.DataFrame(data)

# LabelEncoder nesnesi oluşturuluyor
lbe = LabelEncoder()

# "Type" sütununu LabelEncoder ile dönüştürüyoruz
df["Encoded Type"] = lbe.fit_transform(df["Type"])


# In[38]:


df.head()


# In[39]:


df = pd.read_csv('C:/ai4i2020.csv')
df.head()


# In[40]:


q1 = df["Rotational speed [rpm]"].quantile(0.25)
q3 = df["Rotational speed [rpm]"].quantile(0.75)
iqr = q3-q1
upper_limit = q3 + 1.5* iqr
anormality = df["Rotational speed [rpm]"] > upper_limit
df.loc[anormality,  "Rotational speed [rpm]"] = upper_limit

q1 = df["Torque [Nm]"].quantile(0.25)
q3 = df["Torque [Nm]"].quantile(0.75)
iqr = q3-q1
upper_limit = q3 + iqr * 1.5
lower_limit = q1 - iqr * 1.5
lower_anormality = df["Torque [Nm]"] < lower_limit
upper_anormality = df["Torque [Nm]"] > upper_limit
df.loc[upper_anormality , "Torque [Nm]"] = upper_limit
df.loc[lower_anormality , "Torque [Nm]"] = lower_limit

q1 = df["Tool wear [min]"].quantile(0.25)
q3 = df["Tool wear [min]"].quantile(0.75)
iqr = q3-q1
upper_limit = q3 + iqr * 1.5
anormality = df["Tool wear [min]"] > upper_limit
df.loc[anormality , "Tool wear [min]"] = upper_limit


# In[41]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Örnek bir veri çerçevesi oluşturuyoruz
data = {
    "Type": ["M", "L", "H"]
}

df = pd.DataFrame(data)

# LabelEncoder nesnesi oluşturuluyor
lbe = LabelEncoder()

# "Type" sütununu LabelEncoder ile dönüştürüyoruz
df["Encoded Type"] = lbe.fit_transform(df["Type"])

# One-Hot Encoding ile "Type" sütununu genişletiyoruz
one_hot_encoded = pd.get_dummies(df, columns=["Type"], prefix=["Type"])

print(one_hot_encoded)


# In[42]:


df.head()


# In[43]:


df = pd.read_csv('C:/ai4i2020.csv')
q1 = df["Rotational speed [rpm]"].quantile(0.25)
q3 = df["Rotational speed [rpm]"].quantile(0.75)
iqr = q3-q1
upper_limit = q3 + 1.5* iqr
anormality = df["Rotational speed [rpm]"] > upper_limit
df.loc[anormality,  "Rotational speed [rpm]"] = upper_limit

q1 = df["Torque [Nm]"].quantile(0.25)
q3 = df["Torque [Nm]"].quantile(0.75)
iqr = q3-q1
upper_limit = q3 + iqr * 1.5
lower_limit = q1 - iqr * 1.5
lower_anormality = df["Torque [Nm]"] < lower_limit
upper_anormality = df["Torque [Nm]"] > upper_limit
df.loc[upper_anormality , "Torque [Nm]"] = upper_limit
df.loc[lower_anormality , "Torque [Nm]"] = lower_limit

q1 = df["Tool wear [min]"].quantile(0.25)
q3 = df["Tool wear [min]"].quantile(0.75)
iqr = q3-q1
upper_limit = q3 + iqr * 1.5
anormality = df["Tool wear [min]"] > upper_limit
df.loc[anormality , "Tool wear [min]"] = upper_limit


# In[44]:


from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
df["Encoded Type"] = lbe.fit_transform(df["Type"])


# In[45]:


df.head()


# In[46]:


### Dummy Method / One Hot Method
df_one_hot = pd.get_dummies(df , columns = ["Type"] , prefix = ["Type"])
df[df_one_hot.columns] = df_one_hot


# In[47]:


df.head()


# In[48]:


df["Machine failure"].value_counts()


# In[50]:


plt.figure(figsize=(6, 6))
df["Machine failure"].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel('')
plt.title('Machine failure')
plt.show()
#0'a 1 oranı


# In[51]:


df = df.select_dtypes(include = ['float64' , 'int64','int32'])
df = df.drop('UDI',axis =1)
#UDI bir faydası yok


# In[52]:


from imblearn.over_sampling import SMOTE


# SMOTE'u kullanarak veriyi yeniden örnekleyelim
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Yeniden örneklenmiş verileri içeren DataFrame oluşturalım
df = pd.DataFrame(X_resampled, columns=['Air temperature [K]' , 'Process temperature [K]', 'Rotational speed [rpm]','Torque [Nm]','Encoded Type', 'Tool wear [min]'])
df['Machine failure'] = y_resampled

# Sınıf dağılımını çubuk grafik ile görselleştirelim (yeniden örneklenmiş veri)
plt.figure(figsize=(8, 6))
sns.countplot(x='Machine failure', data=df)
plt.xlabel('Machine Failure')
plt.ylabel('Example Number')
plt.title('Resampling Machine Failure (SMOTE)')
plt.show()


# In[53]:


df.head()


# In[54]:


from sklearn.model_selection import train_test_split


# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak veriyi ayıralım
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']



# Veri setini train ve test setlere bölelim (test_size parametresi test setin yüzdesini belirler)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# # Train setini görüntüleyelim
print("X_train:")
print(X_train)
print("X_test:")
print(y_train)

# # Test setini görüntüleyelim
print("X_test:")
print(X_test)
print("y_test:")
print(y_test)


# In[55]:


df


# In[56]:


pip install seaborn matplotlib


# In[58]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your data is already loaded into a DataFrame called 'df'

# Select the columns you want for the pair plot
columns_for_pairplot = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Encoded Type', 'Tool wear [min]', 'Machine failure']

# Create a pair plot
sns.pairplot(df[columns_for_pairplot])
plt.show()


# In[60]:


from scipy.stats import chi2_contingency
df = pd.read_csv('C:/ai4i2020.csv')
# Assuming your dataset is stored in a pandas DataFrame called 'df'
features = df.drop(['Machine failure','UDI','Product ID','HDF','OSF','PWF','TWF','RNF'], axis=1)  # Exclude the target variable
target = df['Machine failure']
chi2_scores, p_values = [], []
for feature in features.columns:
    contingency_table = pd.crosstab(features[feature], target)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    chi2_scores.append(chi2)
    p_values.append(p)
result = pd.DataFrame({'Feature': features.columns, 'Chi-Square': chi2_scores, 'p-value': p_values})
result = result.sort_values(by='Chi-Square', ascending=False)  # or by 'p-value'
print(result)


# In[ ]:





# In[ ]:




