from scripts.load_data import load_arff_to_dataframe
from scripts.preprocessing import preprocess_dataset
# from scripts.train_model import train_and_evaluate


file_paths = {
    'data/data-class.arff',
    'data/feature-envy.arff',
    'data/god-class.arff',
    'data/long-method.arff',
}

if __name__ == "__main__":
    # Load, preprocess, and train the model for each dataset
    print("hi")
    for file_name in file_paths:
        df = load_arff_to_dataframe(file_name)
        print(df)
        df_clean = preprocess_dataset(df)
        print(df_clean)
        # model_results = train_and_evaluate(df_clean)
        # Save model results to the results directory
