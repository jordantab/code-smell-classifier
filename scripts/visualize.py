from load_data import load_arff_to_dataframe
import matplotlib.pyplot as plt

# Filepaths and their corresponding target variables
filepaths_targets = {
    'data/data-class.arff': 'is_data_class',
    'data/feature-envy.arff': 'is_feature_envy',
    'data/god-class.arff': 'is_god_class',
    'data/long-method.arff': 'is_long_method'
}

for filepath, target_variable in filepaths_targets.items():
    df = load_arff_to_dataframe(filepath)
    print(target_variable)
    null_values = df.isnull().sum()
    null_values = null_values[null_values > 0]
    print(null_values)
    
    # Create a histogram for the target variable
    plt.figure(figsize=(8, 6))
    df[target_variable].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {target_variable.replace("_", " ").title()}')
    plt.xlabel(target_variable)
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()
