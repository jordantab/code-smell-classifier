from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time


def train_random_forest(X_train, X_test, y_train, y_test):
    start_time = time.time()
    
    # Initialize the Random Forest with default parameters
    rf_initial = RandomForestClassifier(random_state=42)
    
    # Train the initial model
    rf_initial.fit(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Conduct grid search cross validation
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=10, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    end_time = time.time()
    training_time = end_time - start_time

    # Compute results
    results = calculate_model_results("Random Forest", y_train, y_test, rf_initial, best_rf, X_train, X_test)

    return (*results, grid_search.best_params_, training_time)

def train_decision_tree(X_train, X_test, y_train, y_test):
    start_time = time.time()

    # Initialize the Decision Tree with default parameters
    dt_initial = DecisionTreeClassifier(random_state=42)
    
    # Train the initial model
    dt_initial.fit(X_train, y_train)
    
    depths = np.arange(10, 21)
    num_leafs = [1, 5, 10, 20, 50, 100]
    param_grid = {
        'criterion':['gini','entropy'],
        'max_depth': depths, 
        'min_samples_leaf': num_leafs
    }


    # Conduct grid search cross validation
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=10, scoring="f1", n_jobs=-1, verbose=1, return_train_score=True)
    grid_search.fit(X_train, y_train)
    best_dt = grid_search.best_estimator_
    end_time = time.time()
    training_time = end_time - start_time

    # Compute results
    results = calculate_model_results("Decision Tree", y_train, y_test, dt_initial, best_dt, X_train, X_test)

    return (*results, grid_search.best_params_, training_time)


def calculate_model_results(model_name, y_train, y_test, model_initial, model_optimized, X_train, X_test):
    # Initial model predictions
    initial_train_pred = model_initial.predict(X_train)
    initial_test_pred = model_initial.predict(X_test)

    # Optimized model predictions
    optimized_train_pred = model_optimized.predict(X_train)
    optimized_test_pred = model_optimized.predict(X_test)

    # Calculate training results
    initial_accuracy_train = accuracy_score(y_train, initial_train_pred)
    optimized_accuracy_train = accuracy_score(y_train, optimized_train_pred)
    initial_f1_train = f1_score(y_train, initial_train_pred, average='weighted')
    optimized_f1_train = f1_score(y_train, optimized_train_pred, average='weighted')

    # print(f"\n{model_name} - Training Results")
    # print("Initial model accuracy: ",initial_accuracy_train)
    # print("Optimized model accuracy: ", optimized_accuracy_train)
    # print("Initial model F1: ", initial_f1_train)
    # print("Optimized model F1: ", optimized_f1_train)

    # Calculate testing results
    initial_accuracy_test = accuracy_score(y_test, initial_test_pred)
    optimized_accuracy_test = accuracy_score(y_test, optimized_test_pred)
    initial_f1_test = f1_score(y_test, initial_test_pred, average='weighted')
    optimized_f1_test = f1_score(y_test, optimized_test_pred, average='weighted')

    # print(f"\n{model_name} - Testing Results")
    # print("Initial model accuracy: ", initial_accuracy_test)
    # print("Optimized model accuracy: ", optimized_accuracy_test)
    # print("Initial model F1: ", initial_f1_test)
    # print("Optimized model F1: ", optimized_f1_test)

    # Return only the testing results
    return (initial_accuracy_train, initial_f1_train, optimized_accuracy_train, optimized_f1_train,
            initial_accuracy_test, initial_f1_test, optimized_accuracy_test, optimized_f1_test)
