# src/train_model.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def evaluate_model(y_pred, y_test):
    return accuracy_score(y_test, y_pred)

#Split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# 2. Train Logistic Regression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


# 3. Predict
def predict(model, X_test):
    return model.predict(X_test)

# 4. Train decision Tree
def train_decision_tree(X_train, y_train):
    model=DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# 5. Random forest 
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,   # number of trees
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# 6. Suppot Vector Machine (SVM)
def train_svm(X_train, y_train):
    model = SVC(kernel='linear')  # start simple
    model.fit(X_train, y_train)
    return model