import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

def plotStrip(x, y, hue, figsize = (14, 9)):
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style("ticks"):
        ax = sns.stripplot(x, y, hue = hue, jitter = 0.4, marker = '.', size = 4, palette = colours)
        ax.set_xlabel('')
        ax.set_xticklabels(["genuine", "fraudulent"], size = 16)
        for axis in ["top","bottom","left","right"]:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ["Transfer", "Cash out"], bbox_to_anchor=(1, 1), loc=2, borderaxespad=0, fontsize = 16)
    return ax

def dispersion_over_time(X, Y):
    '''
    Din grafic se poate observa ca tranzactiile fraudulente is distribuite mai omogen comparat cu cele reale. 
    CASH_OUT is mai multe decat TRANSFER in tranzactii reale, in contrast cu distributia echilibarta
    intre ele in tranzactiile fraudulente.  
    '''
    limit = len(X)
    ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
    ax.set_ylabel("Timp [ore]", size = 16)
    ax.set_title("Amprentele digitale in dungi vs. omogene ale tranzactiilor reale si frauduloase de-a lungul timpului", size = 20)
    plt.show()

def dispersion_over_amount(X, Y):
    # prezenta fraudei in tranzactii poate fi distinsa si din graficul pentru coloana amount
    limit = len(X)
    ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
    ax.set_ylabel("amount", size = 16)
    ax.set_title("Amprentele digitale identice ale tranzactiilor autentice si frauduloase prin suma", size = 18)
    plt.show()

def dispersion_over_errorBalanceDest(X, Y):
    # noua coloana errorBalanceDest este mai eficienta la efectuarea distinctiei dintre tranzactiile reale si frauduloase
    limit = len(X)
    ax = plotStrip(Y[:limit], - X.errorBalanceDest[:limit], X.type[:limit], figsize = (14, 9))
    ax.set_ylabel("-errorBalanceDest", size = 16)
    ax.set_title("Amprentele de polaritate opusa a eroarea din soldurile contului de destinatie", size = 18)
    plt.show()

def separate_transactions(X, Y):
    '''
    graficul 3D distinge cel mai bine intre data frauda si non-frauda folosind coloana bazata pe eroare. 
    Este clar ca coloana step este inefectiva in separarea fraudelor, din cauza naturii dungelor din 
    graficul cu tranzactiile reale vs timp.
    '''
    # compilare destul de lunga
    x = "errorBalanceDest"
    y = "step"
    z = "errorBalanceOrig"
    z_offset = 0.02
    limit = len(X)

    sns.reset_orig() # previne seaborn de la over-riding a mplot3d

    fig = plt.figure(figsize = (10, 12))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X.loc[Y == 0, x][:limit], X.loc[Y == 0, y][:limit], \
    -np.log10(X.loc[Y == 0, z][:limit] + z_offset), c = 'g', marker = '.', \
    s = 1, label = "reale")
        
    ax.scatter(X.loc[Y == 1, x][:limit], X.loc[Y == 1, y][:limit], \
    -np.log10(X.loc[Y == 1, z][:limit] + z_offset), c = 'r', marker = '.', \
    s = 1, label = "fraudulente")

    ax.set_xlabel(x, size = 16); 
    ax.set_ylabel(y + "[ora]", size = 16); 
    ax.set_zlabel("-log$_{10}$ (" + z + ')', size = 16)
    ax.set_title("Coloanele error-based separa tranzactiile reale de cele fraudulente", size = 20)

    plt.axis("tight")
    ax.grid(1)

    no_fraud_marker = mlines.Line2D([], [], linewidth = 0, color='g', marker='.', markersize = 10, label="reale")
    fraud_marker = mlines.Line2D([], [], linewidth = 0, color='r', marker='.', markersize = 10, label="fraudulente")

    plt.legend(handles = [no_fraud_marker, fraud_marker], bbox_to_anchor = (1.20, 0.38 ), frameon = False, prop={"size": 16})
    plt.show()

def fingerprint_transactions(X, Y):
    '''
    dovezi complete incorporate in setul de date ale diferentei dintre tranzactiile frauduloase si autentice sunt 
    obtinute prin examinarea corelatiilor lor respective in hartile termice
    '''
    # facem update la X_fraud & X_non_Fraud cu data curatita 
    X_fraud = X.loc[Y == 1] 
    X_non_fraud = X.loc[Y == 0]
                    
    correlation_non_fraud = X_non_fraud.loc[:, X.columns != "step"].corr()
    mask = np.zeros_like(correlation_non_fraud)
    indices = np.triu_indices_from(correlation_non_fraud)
    mask[indices] = True

    grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
    f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize = (14, 9))

    cmap = sns.diverging_palette(220, 8, as_cmap=True)
    ax1 =sns.heatmap(correlation_non_fraud, ax = ax1, vmin = -1, vmax = 1, cmap = cmap, \
                    square = False, linewidths = 0.5, mask = mask, cbar = False)
    ax1.set_xticklabels(ax1.get_xticklabels(), size = 16); 
    ax1.set_yticklabels(ax1.get_yticklabels(), size = 16); 
    ax1.set_title("Tranzactii \n reale", size = 20)

    correlationFraud = X_fraud.loc[:, X.columns != "step"].corr()
    ax2 = sns.heatmap(correlationFraud, vmin = -1, vmax = 1, cmap = cmap, ax = ax2, \
                    square = False, linewidths = 0.5, mask = mask, yticklabels = False, \
                    cbar_ax = cbar_ax, cbar_kws={"orientation": "vertical", "ticks": [-1, -0.5, 0, 0.5, 1]})
    ax2.set_xticklabels(ax2.get_xticklabels(), size = 16); 
    ax2.set_title("Tranzactii \n frauduloase", size = 20)

    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 14)
    plt.show()