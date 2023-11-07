import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import argparse
import mlflow
import pickle

def read_data(path):
    with open(path,'rb') as f:
        (X_train, X_test, y_train, y_test) = pickle.load(f)
    return (X_train, X_test, y_train, y_test)


def train_model(reg_rate, X_train, y_train):
    model = LogisticRegression(C=1/reg_rate, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    print("training accuracy: ", acc)
    mlflow.log_metric("training accuracy",acc)
    return model

def evaluate_model(model, X_test, y_test):
    #accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test accuracy", acc)
    print("testing accuracy: ", acc)
    #precision
    precision = precision_score(y_test, y_pred)
    print("testing precision: ", precision)
    mlflow.log_metric("test precision", precision)
    #recall
    recall = recall_score(y_test, y_pred)
    print("testing recall: ", recall)
    mlflow.log_metric("test recall", recall)
    #f1-score
    f1_score_ = f1_score(y_test, y_pred)
    print("testing f1-score: ", f1_score_)
    mlflow.log_metric("test f1-score", f1_score_)
    #confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    print("confusion matrix: ", conf_mat)
    #mlflow.log_metric("confusion matrix", conf_mat)
    #auc score
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print("AUC: ", auc)
    mlflow.log_metric("AUC", auc)
    #roc-auc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    plt.plot([0, 1], [0, 1], 'k--') # Plot the diagonal 50% line
    plt.plot(fpr, tpr) # Plot the FPR and TPR achieved by our model
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("ROC-Curve.png")
    mlflow.log_artifact("ROC-Curve.png")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed_data", dest='preprocessed_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', default=0.01, type=float)
    parser.add_argument("--trained_model", dest='trained_model', type=str)
    args = parser.parse_args()
    return args

def main(args):
    (X_train, X_test, y_train, y_test) = read_data(args.preprocessed_data)
    print("training the model")
    model = train_model(args.reg_rate, X_train, y_train)
    mlflow.sklearn.save_model(model, args.model_output)
    print("#"*50)
    print("evaluating the model")
    evaluate_model(model, X_test, y_test)
    print("#"*50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
