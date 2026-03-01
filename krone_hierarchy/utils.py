import pandas as pd

from typing import Dict, List, Set, Tuple
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import ast
def test_metrics(predictions, labels):
    auc = round(roc_auc_score(labels, predictions), 4)
    P = round(precision_score(labels, predictions), 4)
    R = round(recall_score(labels, predictions), 4)
    F1 = round(f1_score(labels, predictions), 4)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    metrics = {"auc": auc, "f1": F1,"p": P,"r": R, "tp": tp, "tn":tn, "fp": fp, "fn": fn}
    return metrics
#
def str_list_to_list_string(str_list: List[str]) -> str:
    return "["+','.join(str_list)+"]"

def is_list_of_strings(column):
    first_element = column.iloc[0]  # Get the first element in the column

    # Check if the first element is a list
    if isinstance(first_element, list):
        # Check if all elements in the list are strings
        return all(isinstance(i, str) for i in first_element)
    return False

def is_list_string(column):
    first_element = column.iloc[0]  # Get the first element in the column
    if isinstance(first_element, str):
        if first_element.startswith("[") and first_element.endswith("]"):
            return True
    return False

def is_string_contain_comma(column):
    return column.apply(lambda x: isinstance(x, str) and ',' in x).any()

def dataframe_string_process(dataframe: pd.DataFrame):
    '''CONVERT LIST TO LIST STRING'''
    string_list_cols = []
    list_string_cols = []
    for col in dataframe.columns:
        if is_list_of_strings(dataframe[col]):
            string_list_cols.append(col)
        if is_string_contain_comma(dataframe[col]):

            list_string_cols.append(col)

    for col in string_list_cols:
        dataframe[col] = dataframe[col].apply(str_list_to_list_string)
    for col in list_string_cols:
        dataframe[col] = '"' + dataframe[col] + '"'

    return dataframe

def dataframe_reverse_string_process(dataframe: pd.DataFrame):
    '''CONVERT LIST STRING TO LIST'''
    list_string_cols = []
    for col in dataframe.columns:
        if is_list_string(dataframe[col]):
            list_string_cols.append(col)

    for col in list_string_cols:
        dataframe[col] = dataframe[col].map(lambda x: x.replace("[","").replace("]","").split(","))

    for col in dataframe.columns:
        if isinstance(dataframe[col].iloc[0], str):
            dataframe[col] = dataframe[col].apply(lambda x: x.replace('"',""))

    return dataframe


if __name__ == '__main__':
    data = {
        'column_1': [['apple', 'cherry', 'fig']],
        'column_2': ['apple,banana'],
    }
    data = pd.DataFrame(data)
    dataframe_string_process(data)
    data.to_csv("data.csv")

    reversed_data = dataframe_reverse_string_process(data)
    print(reversed_data)



