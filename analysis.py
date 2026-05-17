import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

print("Program started...\n")
data = pd.read_csv("students.csv")

print("\n--- Dataset Preview ---")
print(data.head())


print("\n--- Basic Statistics ---")
print("Average Marks:", np.mean(data['Marks']))
print("Maximum Marks:", np.max(data['Marks']))
print("Minimum Marks:", np.min(data['Marks']))


plt.figure()
plt.scatter(data['Study_Hours'], data['Marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()


plt.figure()
plt.scatter(data['Attendance'], data['Marks'])
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.title("Attendance vs Marks")
plt.show()


plt.figure()
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


X = data[['Study_Hours']]
y = data['Marks']

model = LinearRegression()
model.fit(X, y)

study_input = 5
new_data = pd.DataFrame({'Study_Hours': [study_input]})
predicted_marks = model.predict(new_data)

print(f"\nPredicted Marks for {study_input} study hours:", predicted_marks[0])


def assign_grade(marks):
    if marks >= 75:
        return 'A'
    elif marks >= 60:
        return 'B'
    elif marks >= 50:
        return 'C'
    else:
        return 'Fail'

data['Grade'] = data['Marks'].apply(assign_grade)

print("\n--- Dataset with Grades ---")
print(data)


plt.figure()
plt.bar(data['Student'], data['Marks'])
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Marks Distribution")
plt.xticks(rotation=45)
plt.show()
input("\nPress Enter to exit...")