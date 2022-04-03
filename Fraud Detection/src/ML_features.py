import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance, to_graphviz, plot_tree

def features_importance(clf):
    '''
    errorBalanceOrig este cea mai relevanta coloana pentru mode
    coloanele sunt ordonate bazandu-se pe numarul de exemple afectate de split-urile pe acele coloane
    '''
    fig = plt.figure(figsize = (14, 9))
    ax = fig.add_subplot(111)

    colours = plt.cm.Set1(np.linspace(0, 1, 9))

    ax = plot_importance(clf, height = 1, color = colours, grid = False, \
                        show_values = False, importance_type = "cover", ax = ax)
    for axis in ["top","bottom","left","right"]:
        ax.spines[axis].set_linewidth(2)
            
    ax.set_xlabel("scor de importanta", size = 16)
    ax.set_ylabel("Caracteristici/coloane", size = 16)
    ax.set_yticklabels(ax.get_yticklabels(), size = 12)
    ax.set_title("Ordonarea caracteristicilor in functie de importanțt modelului invatat", size = 20)
    plt.show()

def bias_variance_tradeoff(train_sizes, train_scores, cross_val_scores):
    trainScoresMean = np.mean(train_scores, axis=1)
    trainScoresStd = np.std(train_scores, axis=1)
    crossValScoresMean = np.mean(cross_val_scores, axis=1)
    crossValScoresStd = np.std(cross_val_scores, axis=1)

    colours = plt.cm.tab10(np.linspace(0, 1, 9))

    fig = plt.figure(figsize = (14, 9))
    plt.fill_between(train_sizes, trainScoresMean - trainScoresStd,
        trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])
    plt.fill_between(train_sizes, crossValScoresMean - crossValScoresStd,
        crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])
    plt.plot(train_sizes, train_scores.mean(axis = 1), "o-", label = "train", color = colours[0])
    plt.plot(train_sizes, cross_val_scores.mean(axis = 1), "o-", label = "cross-val", color = colours[1])

    ax = plt.gca()
    for axis in ["top","bottom","left","right"]:
        ax.spines[axis].set_linewidth(2)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, ["train", "cross-val"], bbox_to_anchor=(0.8, 0.15), loc=2, borderaxespad=0, fontsize = 16)
    plt.xlabel("Marimea setului de antrenare", size = 16); 
    plt.ylabel("AUPRC", size = 16)
    plt.title("Curba de invatare indica modelul putin underfit", size = 20) 
    plt.show()