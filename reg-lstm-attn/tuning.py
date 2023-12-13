import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

"""
File to facilitate easier graphing using interactive backend outside of Jupyter
"""

def rf_nest_graph(parameters, errors):
    n_estimators = dict()
    for hyperparameters, oob_error in zip(parameters, errors):
        if f'n={hyperparameters["n_estimators"]}' not in n_estimators:
            n_estimators[f'n={hyperparameters["n_estimators"]}'] = list()
        n_estimators[f'n={hyperparameters["n_estimators"]}'].append(oob_error)
    
    fig, ax = plt.figure(), plt.axes()
    stripplot = sns.stripplot(data=pd.DataFrame(n_estimators), legend=True, orient='h', palette='Set2', ax=ax, zorder=0)
    boxplot = sns.boxplot(data=pd.DataFrame(n_estimators), showfliers=False, medianprops={'visible': False}, palette='Set2', orient='h')
    for i, patch in enumerate(boxplot.patches):
        colors = patch.get_facecolor()
        patch.set_facecolor((*colors[:3], 0.3))
    ax.set_xlabel('OOB Error (MSE)')
    ax.set_ylabel('Number of trees (n)')
    ax.set_title('Distribution of Random Forest OOB Error for Various Number of Estimators')
    ax.legend(loc='lower right')
    ax.grid(linestyle=':')
    fig.tight_layout()
    fig.show()
    None

def rf_tree_parameters_graph(parameters, errors, optimal):
    maxFeats, maxDepths, minLeafSamples, oob_color = list(), list(), list(), dict()
    for hyperparameters, oob_error in zip(parameters, errors):
        if hyperparameters['n_estimators'] == optimal['n_estimators']:
            maxFeats.append(hyperparameters['max_features'])
            maxDepths.append(hyperparameters['max_depth'])
            minLeafSamples.append(hyperparameters['min_samples_leaf'])
            oob_color[(hyperparameters['max_features'], hyperparameters['max_depth'], hyperparameters['min_samples_leaf'])] = oob_error

    fig, ax = plt.figure(), plt.subplot(projection='3d')
    vmin, vmax = min(oob_color.values()), max(oob_color.values())
    for maxFeat, maxDepth, minLeafSample in zip(maxFeats, maxDepths, minLeafSamples):
        scatter = ax.scatter(maxFeat, maxDepth, minLeafSample, c=oob_color[(maxFeat, maxDepth, minLeafSample)], cmap='spring', vmin=vmin, vmax=vmax)
    op = ax.scatter(optimal['max_features'], optimal['max_depth'], optimal['min_samples_leaf'], color='deepskyblue', label='Optimal Hyperparameters')
    ax.legend(loc='upper right', handles=[op])
    ax.set_xlabel('Max Feature')
    ax.set_ylabel('Max Depth')
    ax.set_zlabel('Min Leaf Sample')
    fig.suptitle(f'OOB Error for Random Forest Regression with n_estimators={optimal["n_estimators"]}')
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    fig.show()
    None
    

def xgb_parameters_graph(parameters, errors, optimal, lr):
    n_estimators, maxDepths, minChildWeights, loss_color = list(), list(), list(), dict()
    for hyperparameters, kfold_error in zip(parameters, errors):
        n_estimators.append(hyperparameters['n_estimators'])
        maxDepths.append(hyperparameters['max_depth'])
        minChildWeights.append(hyperparameters['min_child_weight'])
        loss_color[(hyperparameters['n_estimators'], hyperparameters['max_depth'], hyperparameters['min_child_weight'])] = kfold_error
            
    fig, ax = plt.figure(), plt.subplot(projection='3d')
    vmin, vmax = min(loss_color.values()), max(loss_color.values())
    for n, maxDepth, minChildWeight in zip(n_estimators, maxDepths, minChildWeights):
        scatter = ax.scatter(maxDepth, minChildWeight, n, c=loss_color[(n, maxDepth, minChildWeight)], cmap='cool', vmin=vmin, vmax=vmax)
    op = ax.scatter(optimal['max_depth'], optimal['min_child_weight'], optimal['n_estimators'], color='orangered', label='Optimal Hyperparameters')
    ax.legend(loc='upper right', handles=[op])
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Min Child Weight')
    ax.set_zlabel('Number of Boosting Iterations')
    fig.suptitle(f'K-Fold Validation Loss (MSE) for Gradient Boosting Trees with {lr} Learning Rate')
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    fig.show()
    None


def main():
    # random forest graphs
    rf_parameters, oob_errors, optimal_rf = np.load('rf_parameters.npy', allow_pickle=True)
    rf_nest_graph(parameters=rf_parameters, errors=oob_errors)
    rf_tree_parameters_graph(parameters=rf_parameters, errors=oob_errors, optimal=optimal_rf)
    
    # xgb graphs
    xgb_parameters, kfold_loss, lr, optimal_xgb = np.load('xgb_parameters.npy', allow_pickle=True)
    xgb_parameters_graph(parameters=xgb_parameters, errors=kfold_loss, optimal=optimal_xgb, lr=lr)


if __name__ == '__main__':
    main()
