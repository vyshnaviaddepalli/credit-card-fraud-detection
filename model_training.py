from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def train_models(X_train, y_train):
    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    models['Decision Tree'] = dt

    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    print("Models trained successfully.")
    return models
