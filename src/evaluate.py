from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def evaluate_classification(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)