import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

# Generate  dataset
n = 1000
df = pd.DataFrame({
    'PatientID': np.random.randint(10000, 99999, size=n),
    'ScheduledDay': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 60, size=n), unit='D'),
    'AppointmentDay': pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(1, 70, size=n), unit='D'),
    'Age': np.random.randint(0, 100, size=n),
    'Gender': np.random.choice(['M', 'F'], size=n),
    'SMS_received': np.random.choice([0, 1], size=n),
    'No-show': np.random.choice(['Yes', 'No'], size=n, p=[0.3, 0.7])
})

# Ensure AppointmentDay is not before ScheduledDay
df['AppointmentDay'] = df.apply(
    lambda x: x['AppointmentDay'] if x['AppointmentDay'] >= x['ScheduledDay']
    else x['ScheduledDay'] + pd.Timedelta(days=1), axis=1
)
df['WaitDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['DayOfWeek'] = df['AppointmentDay'].dt.day_name()
df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

# Optional: Drop unneeded columns
df.drop(['PatientID', 'ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Gender', 'DayOfWeek'], drop_first=True)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Define features and target
X = df.drop('No-show', axis=1)
y = df['No-show']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
import seaborn as sns
import matplotlib.pyplot as plt

# SMS Reminder Effect
sns.barplot(x='SMS_received', y='No-show', data=df)
plt.title("No-show Rate vs SMS Reminder")
plt.show()

# Age vs No-show
sns.histplot(data=df, x='Age', hue='No-show', bins=20, kde=True)
plt.title("Age Distribution by No-show")
plt.show()

# WaitDays vs No-show
sns.boxplot(x='No-show', y='WaitDays', data=df)
plt.title("Wait Time vs No-show")
plt.show()
# Export cleaned data
df.to_csv("cleaned_appointments.csv", index=False)
