import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

from collections import defaultdict

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

