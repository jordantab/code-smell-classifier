from scripts.load_data import load_arff_to_dataframe
from scripts.preprocessing import preprocess_dataset
from scripts.train_model import train_random_forest, train_decision_tree


file_paths = {
    'data/data-class.arff',
    'data/feature-envy.arff',
    'data/god-class.arff',
    'data/long-method.arff',
}

if __name__ == "__main__":
    # Load, preprocess, and train the model for each dataset
    for file_name in file_paths:
        print("\nCode Smell: " + file_name + "\n")
        df = load_arff_to_dataframe(file_name)
        X_train, X_test, y_train, y_test = preprocess_dataset(df)

        # Random forest
        inital_rf, best_rf, initial_accuracy, initial_f1, rf_accuracy, rf_f1, rf_best_params_ = train_random_forest(X_train, X_test, y_train, y_test)
        
        # Decision Tree
        inital_dt, best_dt, dt_inital_accuract, dt_initial_f1, dt_accuracy, dt_f1, dt_best_params_ = train_decision_tree(X_train, X_test, y_train, y_test)
        # Save model results to the results directory
