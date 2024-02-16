from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

def load_arff_to_dataframe(filepath):
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    # Decode byte strings to regular strings for object dtype columns
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df
