
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve


from sklearn.inspection import permutation_importance





def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].set_title(title, fontsize=16)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid(visible=True)
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid(visible=True)
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model", fontsize=16)

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid(visible=True)
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model", fontsize=16)

    return plt








def evaluation(X, y, model, plot_each_model=False, plot_hist_score=False, score='f1-score'):
    
    # Encode target variable y
    le = LabelEncoder()
    le.fit(y)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    if plot_each_model:
        print(f'Corresponding classes to labels : {le_name_mapping}\n')
    y = le.transform(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)

    # Fit model and predict with test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion matrix
    #print(confusion_matrix(y_test, y_pred))
    df_confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred), index=le.classes_, columns=le.classes_)
    df_confusion_matrix.index.name = 'true/pred'
    if plot_each_model:
        print('Confusion matrix :')
        display(df_confusion_matrix)
        print()
    
    # Classification report
    #print(classification_report(y_test, y_pred))
    pd.options.display.float_format = '{:.2f}'.format
    df_classification_report = pd.DataFrame(classification_report(y_test, y_pred, target_names=le_name_mapping.keys(), output_dict=True)).transpose()
    if plot_each_model:
        print('Classification report :')
        display(df_classification_report)
        print()
    
    curr_score = df_classification_report.loc[le_name_mapping.keys(), score]
    #if plot_hist_score:
    #    curr_score = df_classification_report.loc[le_name_mapping.keys(), score]
    #else:
    #    curr_score = None
    
    
    # Feature importance
    dict_feature_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)
    #df_feature_importance = pd.Series(dict_feature_importance.importances_mean, index=model.feature_names_in_)
    df_feature_importance = pd.DataFrame({'Feature name': model.feature_names_in_, 'Feature importance score': dict_feature_importance.importances_mean}).sort_values(by='Feature importance score', axis=0, ascending=False)
    
    if plot_each_model:
        # Plots
        title = f'Learning Curves ({type(model).__name__})'
        fig, axes = plt.subplots(4, 1, figsize=(8, 24))
        fig.subplots_adjust(hspace=0.3)
        plot_learning_curve(
            model, title, X, y, axes=axes, ylim=(-0.01,1.1), cv=4, n_jobs=4
        )
        df_feature_importance.plot.bar(x='Feature name', y='Feature importance score', yerr=dict_feature_importance.importances_std, ax=axes[3])
        axes[3].set_title('Feature importances using permutation on full model')
        axes[3].set_ylabel('Mean accuracy decrease')
        plt.show()
    
 
    return curr_score, df_feature_importance










def pipeline_classification(X, y, dict_models={}, plot_each_model=False, plot_hist_score=False, score='f1-score'):
    
    t0 = time.time()
    
    # Assert input arguments
    if plot_hist_score:
        # assert score != None
        assert not isinstance(score, type(None)),\
            'If <plot_hist_score> is set to True, <score> has to be provided'
        assert score in ['precision', 'recall', 'f1-score'],\
            '<score> argument has to be one of ["precision", "recall", "f1-score"]'
    
    
    
    ###################################################
    def print_header(model):
        nb_pre = (150 - (2 + len(str(model))))//2
        nb_suf = 150 - (nb_pre + len(str(model)) + 2)
        header = (nb_pre - 1) * '#' + (len(str(model)) + 2) * '-' + nb_suf * '#' + '\n'
        header += (nb_pre - 2) * '#' + '| ' + str(model) + ' |' + (nb_suf - 1) * '#' + '\n'
        header += (nb_pre - 1) * '#' + (len(str(model)) + 2) * '-' + nb_suf * '#' + '\n'
        print(header)
    ###################################################
    
    
    
    df_scores = pd.DataFrame()
    dict_df_feature_importance = {}
    
    for model_str in dict_models:

        if plot_each_model:
            print_header(model_str)

        curr_score, df_feature_importance = evaluation(X=X, y=y, model=dict_models[model_str], plot_each_model=plot_each_model, plot_hist_score=plot_hist_score, score=score)

        df_scores[model_str] = curr_score
        dict_df_feature_importance[model_str] = df_feature_importance

        if plot_each_model:
            print(2 * '\n')
    print(150 * '#', '\n')

    
    ###################################################
    def _plot_hist_score(df_scores, target_name):
        
        df_scores_melt = pd.melt(df_scores.reset_index(), id_vars='index', var_name='model', value_name='score').rename(columns={'index': target_name})

        plt.figure(figsize=(len(dict_models)* 2, 6))
        ax = sns.barplot(x='model', y='score', hue=target_name, data=df_scores_melt)
        for i in range(df_scores.shape[0]):
            ax.bar_label(ax.containers[i], fmt='%.2f')
        ax.set(ylim=(0, 1))
        ax.set_title(f'{score} for each model', fontsize=16)
        plt.xticks(rotation=45);
        plt.show()
    ###################################################

    
    if plot_hist_score:
        target_name = y.name
        _plot_hist_score(df_scores, target_name)
        print('\n', 150 * '#')
        
        
        
        
    print(f'\nTime to compute : {time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - t0))}')
    print('\n', 150 * '#')
    print(3 * '\n')
    
    
    return df_scores, dict_df_feature_importance
    
    
    
    
    
    
    
    