import pandas as pd
df = pd.read_csv(r"C:\Users\Administrator\project\heart.csv")

df.shape
# Imbalance Check
class_counts = df['HeartDisease'].value_counts()
print("Class Distribution:\n", class_counts)

# handling imbalance
from imblearn.over_sampling import SMOTE
# Identify categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Perform one-hot encoding for categorical columns
data_encoded = pd.get_dummies(df, columns=categorical_cols)

# Separate features and target variable
X = data_encoded.drop('HeartDisease', axis=1)
y = data_encoded['HeartDisease']

# Apply SMOTE oversampling
oversampler = SMOTE(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

#  Verify the class distribution after oversampling
class_counts = pd.Series(y_resampled).value_counts()
print("Class Distribution after SMOTE:\n", class_counts)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix



#  Separate features and target variable
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Encode categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
encoder = LabelEncoder()
for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])

#  Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Apply feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Apply SMOTE oversampling on the training set
oversampler = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_scaled, y_train)

#  Train a random forest classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = classifier.predict(X_test_scaled)

#  Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_mat)

from sklearn.metrics import accuracy_score, classification_report
classification_rep = classification_report(y_test, y_pred)

print('classification report:\n ',classification_rep )


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the first tree in the random forest
plt.figure(figsize=(20, 18))
plot_tree(classifier.estimators_[0], feature_names=df.columns, class_names=['Normal', 'Heart Disease'], filled=True, fontsize = 8)
plt.show()






# REGRESSION


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score



#Separate features and target variable for regression
X_reg = df.drop('HeartDisease', axis=1)
y_reg = df['HeartDisease']

#Identify categorical columns for classification
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

#  Perform label encoding for categorical columns
encoder = LabelEncoder()
for col in categorical_cols:
    X_reg[col] = encoder.fit_transform(X_reg[col])

# Perform feature scaling for regression
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# Split the dataset for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_reg_train, y_reg_train)

# Make predictions on the test set for regression
y_reg_pred = regressor.predict(X_reg_test)

# Evaluate the regression model's performance
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)
print("Regression Model Performance:")
print("Mean Squared Error:", mse)
print("R2-score:", r2)



import pickle
with open('classifier.pkl','wb') as file:
  pickle.dump(classifier,file)







