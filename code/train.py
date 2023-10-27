import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import argparse


def read_data(path):
    df = pd.read_csv(path)
    return df

def split_data(df):
    X = df[df.columns[1:-1]].values
    y = df[df.columns[-1]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, y_train):
    model = LogisticRegression(C=1/reg_rate, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    print("training accuracy: ", acc)
    return model

def evaluate_model(model, X_test, y_test):
    #accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("testing accuracy: ", acc)
    #precision
    precision = precision_score(y_test, y_pred)
    print("testing precision: ", precision)
    #recall
    recall = recall_score(y_test, y_pred)
    print("testing recall: ", recall)
    #f1-score
    f1_score_ = f1_score(y_test, y_pred)
    print("testing f1-score: ", f1_score_)
    #confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    print("confusion matrix: ", conf_mat)
    #auc score
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print("AUC: ", auc)
    #roc-auc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--') # Plot the diagonal 50% line
    plt.plot(fpr, tpr) # Plot the FPR and TPR achieved by our model
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC-Curve.png")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', default=0.01, type=str)
    args = parser.parse_args()
    return args

def main(args):
    print("processing data")
    df = read_data(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    print("#"*50)
    print("training the model")
    model = train_model(args.reg_rate, X_train, y_train)
    print("#"*50)
    print("evaluating the model")
    evaluate_model(model, X_test, y_test)
    print("#"*50)


if __name__ == "__main__":
    args = parse_args()
    main(args)











