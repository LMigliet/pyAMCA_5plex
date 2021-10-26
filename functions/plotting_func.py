
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy.polynomial.polynomial as poly
import functions.fitting_func as fitfunc


def plot_training_amplification_curves(train_plot, NMETA):

    fig, ax = plt.subplots(1,5, figsize=(15, 5), dpi=300)
    fig.suptitle(f"Training dataset: Amplification Curves", fontsize=27, weight="bold")

    for i, (cpe_type, df) in enumerate(tqdm(train_plot.groupby('CPE_type'), leave=False)):

        ax[i].set_title(f'bla_{cpe_type.upper()}', fontsize=18, weight='bold', c=f"C{i}")
        curves = df.iloc[:, NMETA:].sample(200).transpose()

        ax[i].plot(curves.index, curves.values, c=f"C{i}")
        ax[i].grid(alpha=0.5)
        ax[i].set_ylim((0, 1.2))

        ax[i].tick_params(axis='x', labelsize=15)
        ax[i].tick_params(axis='y', labelsize=15) 

        if i>0:
            ax[i].set_yticklabels([])

        curves2 = train_plot[train_plot.CPE_type==cpe_type].iloc[:, NMETA:].sample(50, replace=True)
        curves2 = curves2.mean()
        ax[i].plot(curves2.index[:-1], curves2.values[:-1], c="black", linewidth=7 , zorder=4)

    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    plt.xlabel('Cycle', fontsize=20)
    plt.ylabel('Fluorescence', fontsize=20)
    
    plt.tight_layout()
    plt.show()


def plot_training_melting_curves(df_MC_filt, NMETA):

    train_plot = df_MC_filt.loc[df_MC_filt.Conc != "unk"]
    train_plot = train_plot.set_index('CPE_type')

    df_cu = []

    for i, df in train_plot.groupby('CPE_type'):
        curves2 = df.iloc[:, NMETA:].mean()
        df_cu.append(curves2)
        
    df_curve_black = pd.concat(df_cu,axis=1)

    ### plotting ###
    fig, ax = plt.subplots(1,1, figsize=(15, 5), dpi=200, constrained_layout=True)
    fig.suptitle(f"Training dataset: Melting Curves", fontsize=27, weight="bold")

    curves = train_plot.iloc[:, NMETA:].sample(400).transpose()
    curves = curves.rename(columns={'imp':'0', 'kpc':'1', 'ndm':'2', 'oxa48':'3', 'vim':"4"})

    for col in curves:
        ax.plot(curves[col].index, curves[col].values, c=f'C0{col}')

    ax.plot(df_curve_black, c='black', lw=8, zorder=6)    
        
    ax.grid(alpha=0.5)
    ax.set_ylim((0, 0.13))
    ax.set_xlim((75,98))

    ax.set_ylabel('dF/dT')
    ax.set_xlabel('Temperature (C)')    

    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15) 
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.show()


def plot_melting_peaks_distribution(train_plot, NMETA):

    bins = np.histogram(train_plot['MeltPeaks'], bins=200)[1]
    fig, ax = plt.subplots(1, 1, figsize=(13,4), dpi=300, constrained_layout=True)
    fig.suptitle("dPCR: Melting Curves Distributions per Targets", fontsize=20, weight="bold")
      
    for i, (target, df) in enumerate(train_plot.groupby('CPE_type')):  
        sns.histplot(df['MeltPeaks'], bins=bins, color=f"C{i}", label=f"bla_{target.upper()}", ax=ax)

        ax.grid(alpha=0.5)
        ax.set_xlim((75, 98))
        ax.set_xticks(np.arange(75, 98, 2.0))

        ax.set_ylabel("PDF", fontsize=16)
        ax.set_xlabel("Melting Temperature (" + chr(176)+ "C)", fontsize=16)

        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16) 

    ax.legend(title="Target", title_fontsize=14, fontsize=13, borderpad=0.1, handletextpad=0.1)

    plt.show()


def plot_training_dilution(df_training, NMETA):

    coeff_dict = {}
    coeff_dict_sm = {}
    intersection_ct_value = {}

    fig, ax = plt.subplots(3, 2, figsize=(11, 8), dpi=300)
    ax = ax.flatten()

    fig.suptitle("Digital Standard Curves", fontsize=24, weight="bold")

    for i, (target, df) in enumerate(df_training.groupby('CPE_type')):
        
        df_meta = df.iloc[:,:NMETA+1]
        df_curves = df.iloc[:,NMETA+1:].T
        
        cts = []
        positions = []
        
        for conc, df_conc in df_meta.groupby('Conc'):

            if conc != "unk":
                cts.append( fitfunc.compute_cts(df_curves.loc[:,df_conc.index], thresh=0.05).squeeze().tolist() )
                positions.append( int(conc) )

        ax[i].set_title(f'Target: bla_{target.upper()}')
        ax[i].boxplot(cts, positions=positions, widths=0.3*np.array(positions), patch_artist=True, showfliers=False)
        ax[i].set_xscale('log')
        
        # start of the X axis
        a, b = 1E0, 1E6
        ax[i].set_xlim((a, b))
        ylims = 13, 27
        ax[i].set_ylim((ylims[0], ylims[1]+2))
        ax[i].grid(True, which="minor", ls="--")

        
        a_end = 3E3  # start of the grey middle area
        b_beg = 4E2  # end of the grey middle area
        
        ax[i].fill_between([a, a_end], ylims[0], ylims[1]+2, alpha=0.2, color='C0')
        ax[i].text( 10**( (np.log10(b_beg) + np.log10(a))/2 ), 
                   ylims[1]+1, 'Single Molecule', horizontalalignment='center', verticalalignment='center')
        
        ax[i].fill_between([b_beg, b], ylims[0], ylims[1]+2, alpha=0.2, color='C1')
        ax[i].text(10**( (np.log10(a_end) + np.log10(b))/2 ), 
                   ylims[1]+1, 'Bulk', horizontalalignment='center', verticalalignment='center')
        
        ax[i].set_ylabel('Cq')
        ax[i].set_xlabel('copies/panel')
        
        x_high, y_high = np.log10(np.array(positions))[-3:], np.array([np.median(item) for item in cts[-3:]])
        x = np.linspace(*np.log10(ax[i].get_xlim()), num=5)
        coefs = poly.polyfit(x_high, y_high, deg=1)
        y = poly.polyval(x, coefs)
        
        coeff_dict[target] = coefs
        
        ax[i].plot(10**x, y, color='C1', lw=2, linestyle='--', zorder=10)
        
        x_high, y_high = np.log10(np.array(positions))[:2], np.array([np.median(item) for item in cts[:2]])
        x = np.linspace(*np.log10(ax[i].get_xlim()), num=5)
        coefs2 = poly.polyfit(x_high, y_high, deg=0)
        y = poly.polyval(x, coefs2)
        ax[i].plot(10**x, y, color='C0', lw=2, linestyle='--', zorder=10)
        
        coeff_dict_sm[target] = coefs2
        
        intersection_ct_value[target] = coefs2.item()
        
    ax[-1].set_visible(False)

    plt.tight_layout()
    plt.show()  


def plot_confusion_matrix(y_true, y_pred, classes, ax,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)
    
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    accuracy = 100 * np.mean(y_true==y_pred)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Classification accuracy: {:.2f}%'.format(accuracy),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")



    ax.grid(False)



def plot_custom_confusion_matrix(y_true, y_pred, classes, ax,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    accuracy = 100 * np.mean(y_true==y_pred)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='AMCA method at Sample Level\nClassification accuracy: {:.2f}%'.format(accuracy),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > 0 else "black")

    ax.grid(False)
    
    for im in ax.get_images():
        im.set_clim(0, 1)
