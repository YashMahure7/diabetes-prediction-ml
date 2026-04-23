from src.preprocessing import preprocess_data
from src.train_model import split_data, train_logistic_regression, train_decision_tree,train_svm, predict, evaluate_model
import pandas as pd
from src.evaluate import evaluate_classification
import matplotlib.pyplot as plt

#Preprocess Data
X, y, scaler = preprocess_data('data/diabetes.csv')

#split
X_train, X_test, y_train, y_test=split_data(X,y)

#Train
model=train_logistic_regression(X_train, y_train)

#Predict
y_pred=predict(model, X_test)

accuracy=evaluate_model(y_pred, y_test)
print("Logistic Regression:", accuracy)

# Train Decision Tree
dt_model=train_decision_tree(X_train, y_train)

#predict model
y_pred_dt=predict(dt_model, X_test)

#Evaluate
dt_accuracy=evaluate_model(y_pred_dt, y_test)

print("Decision tree Accuracy: ", dt_accuracy)

from src.train_model import train_random_forest, predict, evaluate_model

# Train Random Forest
rf_model = train_random_forest(X_train, y_train)

# Predict
y_pred_rf = predict(rf_model, X_test)

# Evaluate
rf_accuracy = evaluate_model(y_test, y_pred_rf)

print("Random Forest Accuracy:", rf_accuracy)

# SVM

# Train SVM
svm_model = train_svm(X_train, y_train)

# Predict
y_pred_svm = predict(svm_model, X_test)

# Evaluate
svm_accuracy = evaluate_model(y_test, y_pred_svm)

print("SVM Accuracy:", svm_accuracy)

#Table for all models
log_acc = accuracy   # Logistic Regression
dt_acc = dt_accuracy
rf_acc = rf_accuracy
svm_acc = svm_accuracy

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM'],
    'Accuracy': [log_acc, dt_acc, rf_acc, svm_acc]
})

print(results)

#Prints the best model with best accuracy
best_model = results.loc[results['Accuracy'].idxmax()] #"index of the maximum."

print("\nBest Model:")
print(best_model)


# Example: evaluate Random Forest
evaluate_classification(y_test, y_pred_rf)

#Visualization

models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM']
accuracies = [log_acc, dt_acc, rf_acc, svm_acc]

plt.figure()
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

plt.xticks(rotation=30)

plt.savefig('images/accuracy_plot.png')  # save plot
plt.show()

# New patient data
new_patient = [[
    2,    # Pregnancies
    120,  # Glucose
    70,   # BloodPressure
    20,   # SkinThickness
    85,   # Insulin
    28.5, # BMI
    0.5,  # DiabetesPedigreeFunction
    30    # Age
]]

# Apply same scaler
new_patient_scaled = scaler.transform(new_patient)

# Predict using best model (example: Random Forest)
prediction = rf_model.predict(new_patient_scaled)

# Output result
if prediction[0] == 1:
    print("The patient is likely diabetic")
else:
    print("The patient is not diabetic")




from src.save_model import save_model, load_model

# Save model
save_model(rf_model, 'models/diabetes_model.pkl')

# Save scaler
save_model(scaler, 'models/scaler.pkl')

loaded_model = load_model('models/diabetes_model.pkl')
loaded_scaler = load_model('models/scaler.pkl')
