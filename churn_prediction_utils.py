import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math

import seaborn as sns
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
# plt.style.use('seaborn')

from sklearn.metrics import confusion_matrix


from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from imblearn.over_sampling import SMOTE

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import fbeta_score, make_scorer

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from graphviz import Source
from IPython.display import SVG,display


f_scorer = make_scorer(fbeta_score, beta=1.5)

def get_input_data_matrix(df, categorical_features, numerical_features, scaling_required = True,
          is_training_data= True, dict_vectorizer = None, standard_scalar = None):
    '''
    This function gets dataframe and converts input features into numpy 2D matrix
    and returns values that need to be predicted.
    Returns:
    {
        'X': X,
        'dict_vectorizer' : dict_vectorizer,
        'standard_scalar' : standard_scalar,
        'feature_names' : feature_names
    }
    '''
    categorical_data = df[categorical_features].to_dict(orient='rows')
    numerical_data = df[numerical_features].to_numpy()
    if is_training_data:
        dict_vectorizer = DictVectorizer(sparse=False)
        dict_vectorizer.fit(categorical_data)
        if scaling_required:
            standard_scalar = StandardScaler()
            standard_scalar.fit(numerical_data)
            X_numerical = standard_scalar.transform(numerical_data)
        else:
            X_numerical = numerical_data
    else:
        if scaling_required:
            X_numerical = standard_scalar.transform(numerical_data)
        else:
            X_numerical = numerical_data


    X_categorical = dict_vectorizer.transform(categorical_data)

    input_data_matrix = np.concatenate((X_categorical, X_numerical), axis=1)
    feature_names = dict_vectorizer.feature_names_ + numerical_features

    return {
        'input_data_matrix': input_data_matrix,
        'dict_vectorizer' : dict_vectorizer,
        'standard_scalar' : standard_scalar,
        'feature_names' : feature_names
    }



def calculate_mi(series, churn_column_values):
    '''
    Calculate mutual information score between the pandas series and churn column
    '''
    return mutual_info_score(series, churn_column_values)


def plot_confusion_matrix(y_actual, y_pred, class_labels, model_name):
    fig, axis = plt.subplots(1, 1, figsize=(8,5))
    matrix = confusion_matrix(y_actual, y_pred)
    axes = sns.heatmap(matrix, square= True, annot= True, fmt= 'd',
                       cbar= True, cmap= plt.cm.Blues, ax = axis)

    axes.set_xlabel('Prediction')
    axes.set_ylabel('Actual')

    tick_marks = np.arange(len(class_labels)) + 0.5

    axes.set_xticks(tick_marks)
    axes.set_xticklabels(class_labels, rotation= 0)

    axes.set_yticks(tick_marks)
    axes.set_yticklabels(class_labels, rotation= 0)

    axes.set_title('Confusion Matrix of {}'.format(model_name))

def plot_roc_curve(fpr, tpr, model_name):
    fig, axis = plt.subplots(1, 1, figsize=(8,5))
    fig.suptitle("AUC-ROC curve of {}".format(model_name))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)


def plot_precision_vs_recall(precisions, recalls, model_name):
    fig = plt.figure(figsize= (8,5))
    fig.suptitle("PR curve of {}".format(model_name))
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, model_name):
    fig = plt.figure(figsize= (8,5))
    fig.suptitle("PR curve of {}".format(model_name))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown


def model_summary(model_name, model_obj, X_train_complete, y_train_complete, X_train,
                  y_train, X_val, y_val, cv, feature_names, evaluation_metrics, metrics_collector_map,
                  options = defaultdict(bool)):

    metrics_collector_map['model_name'].append(model_name)
    for metric in evaluation_metrics:
        if metric == 'f1.5':
            cross_validated_metrics_scores = cross_val_score(model_obj, X_train_complete, y_train_complete, cv= cv,
                                                        scoring= f_scorer)
        else:
            cross_validated_metrics_scores = cross_val_score(model_obj, X_train_complete, y_train_complete, cv= cv,
                                                        scoring= metric)
        cross_validated_metrics_score_mean  = np.mean(cross_validated_metrics_scores)
        metrics_collector_map[metric].append(cross_validated_metrics_score_mean)


    y_proba = cross_val_predict(model_obj, X_train_complete, y_train_complete, cv= cv, method="predict_proba")
    y_scores = y_proba[:, 1]
    y_pred = (y_scores > 0.5).astype(int)
    fpr_roc_auc, tpr_roc_auc, thresholds_roc_auc = roc_curve(y_train_complete, y_scores)

    # PR Curve
    precisions_pr_curve, recalls_pr_curve, pr_curve_thresholds = precision_recall_curve(y_train_complete, y_scores)
    plot_precision_vs_recall(precisions_pr_curve, recalls_pr_curve, model_name)
    plot_precision_recall_vs_threshold(precisions_pr_curve, recalls_pr_curve, pr_curve_thresholds, model_name)

    # ROC_AUC curve
    plot_roc_curve(fpr_roc_auc, tpr_roc_auc, model_name)

    # Classification report
    print(classification_report(y_train_complete, y_pred))

    # Confusion matrix
    plot_confusion_matrix(y_train_complete, y_pred, ['Not Churned', 'Churned'], model_name)

    # Analyze coefficients
    model_obj.fit(X_train, y_train)
    if options['feature_importance_available']:
        if options['coefficients']:
            model_coef = model_obj.coef_.ravel()
        else:
            model_coef = model_obj.feature_importances_
        feature_imp = pd.Series(data = model_coef, index= feature_names).sort_values(ascending = False)
        plt.figure(figsize=(10,12))
        plt.title("Feature importances for {}".format(model_name))
        ax = sns.barplot(y = feature_imp.index, x = feature_imp.values, palette="Blues_d", orient='h')

    #plot decision tree
    if options['tree_based']:
        tree_model = model_obj
        if model_name.startswith('RF'): tree_model = model_obj.estimators_[options['estimated_tree']]
        graph = Source(tree.export_graphviz(tree_model, out_file=None, rounded=True, proportion = False,
                                            feature_names = feature_names, precision = 2,
                                            class_names = ["Not churn", "Churn"],
                                            filled = True
                                           )
                      )
        display(graph)

    # Evaluation metrics so far
    display(pd.DataFrame(metrics_collector_map))


def get_input_data_matrix_with_specific_features(X, required_features, all_features):
    indices = []
    for required_feature in required_features:
        indices.append(all_features.index(required_feature))
    return X[:, indices]



def train_and_evaluate_model_with_smote(clone_clf, X_train_folds, y_train_folds, X_test_fold, y_test_fold,
                           model_to_evaluation_metrics_with_smote_map, model_name, evaluation_metrics):

    if model_name not in model_to_evaluation_metrics_with_smote_map:
        model_to_evaluation_metrics_with_smote_map[model_name] = {metric : [] for metric in evaluation_metrics}

    sm = SMOTE(random_state= 7)

    X_train_fold_oversampled, y_train_fold_oversampled = sm.fit_sample(X_train_folds, y_train_folds)

    clone_clf.fit(X_train_fold_oversampled, y_train_fold_oversampled)

    y_proba_fold = clone_clf.predict_proba(X_test_fold)
    y_scores_fold = y_proba_fold[:, 1]
    y_pred_fold = (y_scores_fold > 0.5).astype(int)

    for metric in evaluation_metrics:
        if metric == 'f1.5':
            metric_value = fbeta_score(y_test_fold, y_pred_fold, beta=1.5)
        if metric == 'roc_auc':
            metric_value = roc_auc_score(y_test_fold, y_scores_fold)
        elif metric == 'f1':
            metric_value = f1_score(y_test_fold, y_pred_fold)
        elif metric == 'recall':
            metric_value = recall_score(y_test_fold, y_pred_fold)
        elif metric == 'precision':
            metric_value = precision_score(y_test_fold, y_pred_fold)
        elif metric == 'accuracy':
            metric_value = accuracy_score(y_test_fold, y_pred_fold)

        model_to_evaluation_metrics_with_smote_map[model_name][metric].append(metric_value)
    return y_pred_fold, y_scores_fold


def model_summary_with_smote(model_name, model_info_map, X_train_full_not_scaled,
                             y_train, X_train_full_scaled, model_to_evaluation_metrics_with_smote_map,
                             evaluation_metrics):

    model_obj = model_info_map['definition']
    options  = model_info_map['options']
    if options['scaling_required']:
        X_train = X_train_full_scaled
    else:
        X_train = X_train_full_not_scaled

    model_to_evaluation_metrics_with_smote_map[model_name] = {metric : [] for metric in evaluation_metrics}

    skfolds = StratifiedKFold(n_splits= options['kfold'], random_state=42)

    y_test_all_folds = np.array([])
    y_scores_all_folds = np.array([])
    y_pred_all_folds = np.array([])
    for train_index, test_index in skfolds.split(X_train, y_train):
        clone_clf = clone(model_obj)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train[test_index]

        y_pred_fold, y_scores_fold = train_and_evaluate_model_with_smote(clone_clf, X_train_folds, y_train_folds,
                                                                         X_test_fold, y_test_fold,
                                                                         model_to_evaluation_metrics_with_smote_map,
                                                          model_name, evaluation_metrics)

        y_test_all_folds = np.concatenate([y_test_all_folds, y_test_fold])
        y_scores_all_folds = np.concatenate([y_scores_all_folds, y_scores_fold])
        y_pred_all_folds = np.concatenate([y_pred_all_folds, y_pred_fold])

    fpr_roc_auc, tpr_roc_auc, thresholds_roc_auc = roc_curve(y_test_all_folds, y_scores_all_folds)

    # PR Curve
    precisions_pr_curve, recalls_pr_curve, pr_curve_thresholds = precision_recall_curve(y_test_all_folds, y_scores_all_folds)
    plot_precision_vs_recall(precisions_pr_curve, recalls_pr_curve, model_name)
    plot_precision_recall_vs_threshold(precisions_pr_curve, recalls_pr_curve, pr_curve_thresholds, model_name)

    # ROC_AUC curve
    plot_roc_curve(fpr_roc_auc, tpr_roc_auc, model_name)

    # Classification report
    print(classification_report(y_test_all_folds, y_pred_all_folds))

    # Confusion matrix
    plot_confusion_matrix(y_test_all_folds, y_pred_all_folds, ['Not Churned', 'Churned'], model_name)


def get_smote_model_names_to_model_objects_map():
    smote_model_names_to_model_objects_map = {}
    smote_model_names_to_model_objects_map['LR']  = {
        'definition' : LogisticRegression(solver='liblinear', random_state= 42),
        'options' : {'scaling_required' : True, 'kfold' : 10}
    }

    smote_model_names_to_model_objects_map['DT']  = {
        'definition' : DecisionTreeClassifier(random_state= 7, max_depth = 3),
        'options' : {'scaling_required' : False, 'kfold' : 10}
    }

    smote_model_names_to_model_objects_map['RF']  = {
        'definition' : RandomForestClassifier(n_estimators=200, max_leaf_nodes=16, n_jobs=-1, random_state=7),
        'options' : {'scaling_required' : False,  'kfold' : 10}
    }

    smote_model_names_to_model_objects_map['KNN']  = {
        'definition' : KNeighborsClassifier(),
        'options' : {'scaling_required' : True, 'kfold' : 10}
    }

    smote_model_names_to_model_objects_map['GNB']  = {
        'definition' : KNeighborsClassifier(),
        'options' : {'scaling_required' : False, 'kfold' : 10}
    }

    smote_model_names_to_model_objects_map['LDA']  = {
        'definition' : LinearDiscriminantAnalysis(),
        'options' : {'scaling_required' : False, 'kfold' : 10}
    }
    return smote_model_names_to_model_objects_map

def get_model_to_evaluation_metrics_with_smote_map(smote_model_names_to_model_objects_map, X_train_full_scaled,
    y_train_full, X_train_full_not_scaled, evaluation_metrics):
    model_to_evaluation_metrics_with_smote_map = {}
    for model_name, model_info_map in smote_model_names_to_model_objects_map.items():
        print("Training : {}".format(model_name))
        model_summary_with_smote(model_name, model_info_map, X_train_full_scaled,
                                 y_train_full, X_train_full_not_scaled,
                                 model_to_evaluation_metrics_with_smote_map, evaluation_metrics)
    return model_to_evaluation_metrics_with_smote_map


def get_model_to_mean_evaluation_metrics_with_smote_df(model_to_evaluation_metrics_with_smote_map, evaluation_metrics):
    model_to_mean_evaluation_metrics_with_smote_map = defaultdict(list)
    for model_name, metrics_map in model_to_evaluation_metrics_with_smote_map.items():
        model_to_mean_evaluation_metrics_with_smote_map['model_name'].append(model_name)
        for metric_name in evaluation_metrics:
            value = np.mean(metrics_map[metric_name])
            model_to_mean_evaluation_metrics_with_smote_map[metric_name].append(value)
    return pd.DataFrame(model_to_mean_evaluation_metrics_with_smote_map)


