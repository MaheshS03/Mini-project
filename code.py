import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("international migration1.csv")
df.head(10)
df.info()
df.describe()
df.isnull().sum()
~df.duplicated()
df1=df[~df.duplicated()]
df1
q1=df['estimate'].quantile(0.25)
q3=df['estimate'].quantile(0.75)
IQR=q3-q1
print("First quantile:",q1," Third quantile:",q3," IQR: ",IQR,"\n")
lower=q1-1.5*IQR
upper=q3+1.5*IQR
outliers=df[(df['estimate']>=lower)&(df['estimate']<=upper)]
from scipy.stats import zscore
z=outliers[(zscore(outliers['estimate'])<3)]
print("Cleaned Data: \n")
print(z)
df.skew()
z.skew()
df.kurtosis()
z.kurtosis()
sns.boxplot(x="estimate",data=df)
sns.boxplot(x="estimate",data=z)
sns.countplot(x="no of passengers",data=df)
sns.countplot(x="no of passengers",data=z)
sns.histplot(df["estimate"])
sns.histplot(z["estimate"])
sns.scatterplot(x=df['estimate'],y=df['no of passengers'])
sns.scatterplot(x=z['estimate'],y=z['no of passengers'])
df.corr()
z.corr()
sns.heatmap(df.corr(),annot=True)
sns.heatmap(z.corr(),annot=True)
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['estimate'],fit=True,line='45')
plt.show()
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(z['estimate'],fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['estimate']=pd.DataFrame(transformer.fit_transform(df[['estimate']]))
sm.qqplot(df['estimate'],line='45')
plt.show()
transformer=PowerTransformer("yeo-johnson")
z['estimate']=pd.DataFrame(transformer.fit_transform(z[['estimate']]))
sm.qqplot(z['estimate'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df['standard_error']=pd.DataFrame(qt.fit_transform(df[['standard_error']]))
sm.qqplot(df['standard_error'],line='45')
plt.show()
qt=QuantileTransformer(output_distribution='normal')
z['standard_error']=pd.DataFrame(qt.fit_transform(z[['standard_error']]))
sm.qqplot(z['standard_error'],line='45')
plt.show()
z.drop(["year_month","month_of_release"],axis=1, inplace=True)
z
sns.scatterplot(x="standard_error",y="no of passengers",hue="sex",data=df)
sns.scatterplot(x="standard_error",y="no of passengers",hue="sex",data=z)
sns.histplot(data=df, x="estimate", hue="status", element="step", stat="density")
sns.histplot(data=z, x="estimate", hue="status", element="step", stat="density")
sns.histplot(data=df, x="no of passengers", hue="day of week" , element="step", stat="density")
sns.relplot(data=df,x=df["country"],y=df["estimate"],hue="sex")
plt.xticks(rotation = 90)
plt.show()
sns.relplot(data=z,x=z["country"],y=z["estimate"],hue="sex")
plt.xticks(rotation = 90)
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

selected_columns = ['sex', 'estimate', 'standard_error','no of passengers']
data = df[selected_columns]

X = data.drop('sex', axis=1)
y = data['sex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


sns.histplot(data=z, x="no of passengers", hue="day of week" , element="step", stat="density")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

selected_columns = ['sex', 'estimate', 'standard_error','no of passengers']
data = z[selected_columns]

X = data.drop('sex', axis=1)
y = data['sex']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
df2=z.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
df2['Estimate']=le.fit_transform(df2['estimate'])
df2['SDE']=le.fit_transform(df2['standard_error'])
df2['Passengers']=le.fit_transform(df2['no of passengers'])
df2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Assuming you have your dataset loaded into X (features) and y (labels/targets)
selected_columns = ['sex', 'estimate', 'standard_error','no of passengers']
data = z[selected_columns]

X = data.drop('sex', axis=1)
y = data['sex']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:", nb_accuracy)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Assuming you have your dataset loaded into X (features) and y (labels/targets)
selected_columns = ['sex', 'estimate', 'standard_error','no of passengers']
data = df[selected_columns]

X = data.drop('sex', axis=1)
y = data['sex']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Accuracy:", nb_accuracy)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Assuming you have loaded the dataset into a DataFrame called 'data'

# Select the relevant columns for the linear regression
selected_columns = ['passenger_type', 'direction', 'sex', 'estimate', 'standard_error', 'status', 'day of week', 'country', 'no of passengers']
data_selected = z[selected_columns]

# Split the data into input features (X) and target variable (y)
X = data_selected.drop('estimate', axis=1)
y = data_selected['estimate']

# Perform one-hot encoding on categorical variables
categorical_columns = ['passenger_type', 'direction', 'sex', 'status', 'day of week', 'country']
preprocessor = ColumnTransformer([('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')
X_encoded = preprocessor.fit_transform(X)

# Create an instance of the Linear Regression model
model = LinearRegression()

# Fit the model to the encoded data
model.fit(X_encoded, y)

# Make predictions on the encoded data
predictions = model.predict(X_encoded)

# Calculate the mean squared error
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Print the R-squared value
print("R-squared:", r2)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Assuming you have loaded the dataset into a DataFrame called 'data'

# Select the relevant columns for the linear regression
selected_columns = ['passenger_type', 'direction', 'sex', 'estimate', 'standard_error', 'status', 'day of week', 'country', 'no of passengers']
data_selected = df[selected_columns]

# Split the data into input features (X) and target variable (y)
X = data_selected.drop('estimate', axis=1)
y = data_selected['estimate']

# Perform one-hot encoding on categorical variables
categorical_columns = ['passenger_type', 'direction', 'sex', 'status', 'day of week', 'country']
preprocessor = ColumnTransformer([('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')
X_encoded = preprocessor.fit_transform(X)

# Create an instance of the Linear Regression model
model = LinearRegression()

# Fit the model to the encoded data
model.fit(X_encoded, y)

# Make predictions on the encoded data
predictions = model.predict(X_encoded)

# Calculate the mean squared error
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Print the R-squared value
print("R-squared:", r2)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Select relevant columns for the model
selected_columns = ['estimate', 'standard_error']
X = z[selected_columns]
y = z['no of passengers']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Select relevant columns for the model
selected_columns = [ 'estimate', 'standard_error']
X = df[selected_columns]
y = df['no of passengers']

# Perform data preprocessing (e.g., handle missing values, encode categorical variables) if required

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Select relevant columns for the model
selected_columns = [ 'estimate', 'standard_error']
X = df[selected_columns]
y = df['no of passengers']

# Perform data preprocessing (e.g., handle missing values, encode categorical variables) if required

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
k = 2 # Number of top features to select
selector = SelectKBest(f_regression, k=k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(X_train_selected, y_train)
y_pred = model1.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Select relevant columns for the model
selected_columns = [ 'estimate', 'standard_error']
X = z[selected_columns]
y = z['no of passengers']

# Perform data preprocessing (e.g., handle missing values, encode categorical variables) if required

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
k = 2 # Number of top features to select
selector = SelectKBest(f_regression, k=k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(X_train_selected, y_train)
y_pred = model1.predict(X_test_selected)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)






