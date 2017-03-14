import numpy as np
import pprint
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import re
import sys
from itertools import tee
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn import svm, datasets
import runpy
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.externals import joblib

#--------------------------------------------------------------------------------------------------------------------------------
def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: http://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: http://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - http://stackoverflow.com/a/16124677/395857 
    - http://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on http://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)
#--------------------------------------------------------------------------------------------------------------------------------

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#our window iterator 
#http://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator-in-python
def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)

#our "word" processor	
def wordpro(wordlength):
	z = 0
	if (wordlength == 3):
		z = 1
	if (wordlength == 5):
		z = 2
	if (wordlength == 7):
		z = 3
	if (wordlength == 9):
		z = 4
	if (wordlength == 11):
		z = 5
	if (wordlength == 13):
		z = 6
	if (wordlength == 15):
		z = 7
	if (wordlength == 17):
		z = 8
	if (wordlength == 19):
		z = 9
	if (wordlength == 21):
		z = 10	
	return z
	
#removing all the blank spaces from the file and lines with "topology"
def linebreaker (filename):
	z = open('prototext.txt','w')
	with open(filename) as f:
		for line in f:
			if not line.isspace():
				#sys.stdout.write(line)
					if '>' not in line:
						z.write(line)
	z.close()										

#Creating an amino_acid_word:feature pair using DictVectorizer
#Fixed wordlenghts for optimisation
def matrix (wordlength,length,instance):
	di = {'i': 1, 'o': 0, 'P': 0, 'L':0}
	damino = {'A': 1,'R': 2,'D': 3,'N': 4,'C': 5,'E': 6,'Q': 7,'G': 8,'H': 9,'I': 10,'L': 11,'K': 12,'M': 13,'F': 14,'P': 15,'S': 16,'T': 17,'W': 18,'Y': 19,'V': 20,'J': 21}
	word_list = []
	word_list_w = []
	toplogy_list = []
	toplogy_w = []
	tempd = ''
	z = wordpro(wordlength)
	filein = open('prototext.txt','r')
	for line in filein:
			temp_line = line.rstrip()
			#Adding charachters in the begening and end for windows
			temporary_string = ("J" * z)+(temp_line)+("J" * z)
			for each in window(temporary_string, wordlength):
				temp = ''.join(each)
				temp_c = []
				for c in temp:
					g = damino[c]
					temp_c.append(g)
				word_list.append(temp)
			temporary_topology = next(filein)
			temporary_topology = temporary_topology.rstrip()
			for c in temporary_topology:
				k = di[c]
				toplogy_list.append(k)
				toplogy_w.append(c)
	#http://stackoverflow.com/questions/30522724/take-multiple-lists-into-dataframe
	#http://stackoverflow.com/questions/20970279/how-to-do-a-left-right-and-mid-of-a-string-in-a-pandas-dataframe
	dftemp = pd.DataFrame({'word_list':word_list})
	dwtemp = pd.DataFrame({'word_list':word_list_w})
	if(length == 3):
		df = pd.DataFrame({'p-1':dftemp['word_list'].str[0],'p':dftemp['word_list'].str[1],'p+1':dftemp['word_list'].str[2]})

	if(length == 5):
		df = pd.DataFrame({'p-2':dftemp['word_list'].str[0],'p-1':dftemp['word_list'].str[1],'p':dftemp['word_list'].str[2],'p+1':dftemp['word_list'].str[3],'p+2':dftemp['word_list'].str[4]})

	if(length == 7):
		df = pd.DataFrame({'p-3':dftemp['word_list'].str[0],'p-2':dftemp['word_list'].str[1],'p-1':dftemp['word_list'].str[2],'p':dftemp['word_list'].str[3],'p+1':dftemp['word_list'].str[4],'p+2':dftemp['word_list'].str[5],'p+3':dftemp['word_list'].str[6]})

	if(length == 9):
		df = pd.DataFrame({'p-4':dftemp['word_list'].str[0],'p-3':dftemp['word_list'].str[1],'p-2':dftemp['word_list'].str[2],'p-1':dftemp['word_list'].str[3],'p':dftemp['word_list'].str[4],'p+1':dftemp['word_list'].str[5],'p+2':dftemp['word_list'].str[6],'p+3':dftemp['word_list'].str[7],'p+4':dftemp['word_list'].str[8]})

	if(length == 11):
		df = pd.DataFrame({'p-5':dftemp['word_list'].str[0],'p-4':dftemp['word_list'].str[1],'p-3':dftemp['word_list'].str[2],'p-2':dftemp['word_list'].str[3],'p-1':dftemp['word_list'].str[4],'p':dftemp['word_list'].str[5],'p+1':dftemp['word_list'].str[6],'p+2':dftemp['word_list'].str[7],'p+3':dftemp['word_list'].str[8],'p+4':dftemp['word_list'].str[9],'p+5':dftemp['word_list'].str[10]})

	if(length == 13):
		df = pd.DataFrame({'p-6':dftemp['word_list'].str[0],'p-5':dftemp['word_list'].str[1],'p-4':dftemp['word_list'].str[2],'p-3':dftemp['word_list'].str[3],'p-2':dftemp['word_list'].str[4],'p-1':dftemp['word_list'].str[5],'p':dftemp['word_list'].str[6],'p+1':dftemp['word_list'].str[7],'p+2':dftemp['word_list'].str[8],'p+3':dftemp['word_list'].str[9],'p+4':dftemp['word_list'].str[10],'p+5':dftemp['word_list'].str[11],'p+6':dftemp['word_list'].str[12]})

	if(length == 15):
		df = pd.DataFrame({'p-7':dftemp['word_list'].str[0],'p-6':dftemp['word_list'].str[1],'p-5':dftemp['word_list'].str[2],'p-4':dftemp['word_list'].str[3],'p-3':dftemp['word_list'].str[4],'p-2':dftemp['word_list'].str[5],'p-1':dftemp['word_list'].str[6],'p':dftemp['word_list'].str[7],'p+1':dftemp['word_list'].str[8],'p+2':dftemp['word_list'].str[9],'p+3':dftemp['word_list'].str[10],'p+4':dftemp['word_list'].str[11],'p+5':dftemp['word_list'].str[12],'p+6':dftemp['word_list'].str[13],'p+7':dftemp['word_list'].str[14]})

	if(length == 17):
		df = pd.DataFrame({'p-8':dftemp['word_list'].str[0],'p-7':dftemp['word_list'].str[1],'p-6':dftemp['word_list'].str[2],'p-5':dftemp['word_list'].str[3],'p-4':dftemp['word_list'].str[4],'p-3':dftemp['word_list'].str[5],'p-2':dftemp['word_list'].str[6],'p-1':dftemp['word_list'].str[7],'p':dftemp['word_list'].str[8],'p+1':dftemp['word_list'].str[9],'p+2':dftemp['word_list'].str[10],'p+3':dftemp['word_list'].str[11],'p+4':dftemp['word_list'].str[12],'p+5':dftemp['word_list'].str[13],'p+6':dftemp['word_list'].str[14],'p+7':dftemp['word_list'].str[15],'p+8':dftemp['word_list'].str[16]})
		
	if(length == 19):
		df = pd.DataFrame({'p-9':dftemp['word_list'].str[0],'p-8':dftemp['word_list'].str[1],'p-7':dftemp['word_list'].str[2],'p-6':dftemp['word_list'].str[3],'p-5':dftemp['word_list'].str[4],'p-4':dftemp['word_list'].str[5],'p-3':dftemp['word_list'].str[6],'p-2':dftemp['word_list'].str[7],'p-1':dftemp['word_list'].str[8],'p':dftemp['word_list'].str[9],'p+1':dftemp['word_list'].str[10],'p+2':dftemp['word_list'].str[11],'p+3':dftemp['word_list'].str[12],'p+4':dftemp['word_list'].str[13],'p+5':dftemp['word_list'].str[14],'p+6':dftemp['word_list'].str[15],'p+7':dftemp['word_list'].str[16],'p+8':dftemp['word_list'].str[17],'p+9':dftemp['word_list'].str[18]})
		
	if(length == 21):
		df = pd.DataFrame({'p-10':dftemp['word_list'].str[0],'p-9':dftemp['word_list'].str[1],'p-8':dftemp['word_list'].str[2],'p-7':dftemp['word_list'].str[3],'p-6':dftemp['word_list'].str[4],'p-5':dftemp['word_list'].str[5],'p-4':dftemp['word_list'].str[6],'p-3':dftemp['word_list'].str[7],'p-2':dftemp['word_list'].str[8],'p-1':dftemp['word_list'].str[9],'p':dftemp['word_list'].str[10],'p+1':dftemp['word_list'].str[11],'p+2':dftemp['word_list'].str[12],'p+3':dftemp['word_list'].str[13],'p+4':dftemp['word_list'].str[14],'p+5':dftemp['word_list'].str[15],'p+6':dftemp['word_list'].str[16],'p+7':dftemp['word_list'].str[17],'p+8':dftemp['word_list'].str[18],'p+9':dftemp['word_list'].str[19],'p+10':dftemp['word_list'].str[20]})
	
	train_dict = df.T.to_dict().values()
	#print (train_dict)
	vectorizer = DV( sparse = False )
	vec_train = vectorizer.fit_transform( train_dict )
	print (vectorizer.get_feature_names())
	target = np.asarray(toplogy_list)
	X_train, X_test, y_train, y_test = train_test_split(vec_train, target, test_size=0.2, random_state=0)
	estimator = svm.SVC(kernel='rbf')
	cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
	gammas = np.logspace(-6, -1, 10)
	classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))	
	X_train, X_test, y_train, y_test = train_test_split(vec_train, target, test_size=0.33, random_state=0)
	classifier.fit(vec_train, target)
	classifier_prediction = classifier.predict(X_test)
	print ('\nClasification report i:\n', classification_report(y_test, classifier_prediction))
	plot_classification_report(classification_report(y_test, classifier_prediction))
	plt.savefig('i_plot_classif_report.pdf', dpi=200, format='pdf', bbox_inches='tight')
	plt.close()
	joblib.dump(classifier, 'i.pkl', compress=9)
	

window_size = [21] 
linebreaker ('membrane-beta_4state.3line.txt')
for i in window_size:
	matrix(i,i,i)