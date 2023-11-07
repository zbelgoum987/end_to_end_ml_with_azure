import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import pickle


def read_data(input_path):
    df = pd.read_csv(input_path)
    return df

def split_data(df):
    X = df[df.columns[1:-1]].values
    y = df[df.columns[-1]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def save_output_data(output_path, X_train, X_test, y_train, y_test):
    with open(output_path,"wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test),f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", dest='input_data', type=str)
    parser.add_argument("--output_path", dest='output_path', type=str)
    args = parser.parse_args()
    return args


def main(args):
    df = read_data(args.input_data)
    print("splitting_data")
    print("#"*50)
    X_train, X_test, y_train, y_test = split_data(df)
    print("normlizing data")
    print("#"*50)
    X_train, X_test, y_train, y_test = normalize_data(X_train, X_test, y_train, y_test)
    print("saving preprocessed data")
    save_output_data(args.output_path,X_train, X_test, y_train, y_test)
    print("#"*50)


if __name__ == "__main__":
    args = parse_args()
    main(args)


    