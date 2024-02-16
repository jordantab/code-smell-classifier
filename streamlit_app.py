import streamlit as st
from scripts.load_data import load_arff_to_dataframe
from scripts.preprocessing import preprocess_dataset
from scripts.train_model import train_random_forest, train_decision_tree

# Available data files
file_paths = [
    'data/data-class.arff',
    'data/feature-envy.arff',
    'data/god-class.arff',
    'data/long-method.arff',
]

# User selects the data file
selected_file = st.selectbox('Choose a data file to load:', file_paths)

# User selects the algorithm
algorithm = st.selectbox('Choose an algorithm:', ['Random Forest', 'Decision Tree'])

# Button to load, preprocess, and train the model
if st.button('Load and Train'):
    df = load_arff_to_dataframe(selected_file)
    X_train, X_test, y_train, y_test = preprocess_dataset(df)

    if algorithm == 'Random Forest':
        results = train_random_forest(X_train, X_test, y_train, y_test)
    elif algorithm == 'Decision Tree':
        results = train_decision_tree(X_train, X_test, y_train, y_test)

    initial_accuracy_train, initial_f1_train, optimized_accuracy_train, optimized_f1_train, initial_accuracy_test, initial_f1_test, optimized_accuracy_test, optimized_f1_test, best_params, training_time = results

    # Display comparison
    st.write(f"**Algorithm: {algorithm}**")
    st.write(f"**Code Smell: {selected_file.split('/')[-1]}**")
    st.write("Training Time:", training_time, "seconds")
    st.write("Best Parameters:", best_params)
    st.write("### Metrics Comparison")
    st.write("Metric", "Training (Initial)", "Test (Initial)", "Training (Optimized)", "Test (Optimized)")
    st.write("Accuracy", f"{initial_accuracy_train*100:.2f}%", f"{initial_accuracy_test*100:.2f}%", f"{optimized_accuracy_train*100:.2f}%", f"{optimized_accuracy_test*100:.2f}%")
    st.write("F1-Score", f"{initial_f1_train*100:.2f}%", f"{initial_f1_test*100:.2f}%", f"{optimized_f1_train*100:.2f}%", f"{optimized_f1_test*100:.2f}%")

