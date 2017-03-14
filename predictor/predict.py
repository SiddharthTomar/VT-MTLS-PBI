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
from sklearn.externals import joblib
from sklearn import preprocessing

#PSSM based predictor!!

#Code to read the PSSM and convert the input to a pandas dataframe
def pssmtodf ():
	counter = 0
	store_lines = []
	tempindex = ("pssm.txt")
	file = open (tempindex, 'r')
	lines = file.readlines()
	if (counter == 0):
		lines = lines[2:-4]
		lines[0] = "Index Nucleotide A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V   AA   RR   NN   DD   CC   QQ   EE   GG   HH   II   LL   KK   MM   FF   PP   SS   TT   WW   YY   VV NA NA"
		counter = counter + 1
	for k in lines:
		k = re.sub("\s+", ",", k.strip())
		store_lines.append(k)
		#print (i)
	#File for debugging in case of any problems with input
	outfile = open('pssmfinal-debug.csv', 'w')
	for item in store_lines:
		outfile.write("%s\n" % item)
	outfile.close()
	df = pd.read_csv("pssmfinal-debug.csv")
	keep_cols = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
	df = df[keep_cols]	
	#The window-maker
	vals = df.values
	idx = np.tile(np.arange(21), (len(df) - 21,1)) + np.arange(len(df) - 21).reshape(-1,1)
	#printing the refrence array for debugging - Slightly unpredictable behaviour. Check input if last row/column is less than the sequence lenght
	print (idx)
	cols = [ "{}_{}".format(c,str(i)) for i in range(21) for c in df.columns]
	df2 = pd.DataFrame(vals[idx.flatten()].reshape(len(df)-21,df.shape[1]*21), columns=cols)
	df2.to_csv("csv1.csv", index=False)
	return (df2)

	
df = pssmtodf()
result = []
result_i = []
result_o = []
result_P = []
result_L = []
#Automatic conversion using DictVectorizer
train_dict = df.T.to_dict().values()
vectorizer = DV( sparse = False )
vec_in = vectorizer.fit_transform( train_dict )
#Scaling the input
max_abs_scaler = preprocessing.MaxAbsScaler()
vec_in = max_abs_scaler.fit_transform(vec_in)
#Dictionaries for converting the class
di = {'0': 'i', '1': 'o', '2': 'P', '3':'L'}
di_i = {'1': 'i', '0': '-'}
di_o = {'1': 'o', '0': '-'}
di_P = {'1': 'P', '0': '-'}
di_L = {'1': 'L', '0': '-'}
#---------------------------------------------------
model = joblib.load('All.pkl')
prediction = model.predict(vec_in)
for i in prediction:
	z = str(i)
	temp = di[z]
	result.append(temp)
file = open('output_all.txt', 'w')
for item in result:
	file.write("%s" % item)
#---------------------------------------------------
#Model predicting just i
#---------------------------------------------------
model = joblib.load('i.pkl')
prediction = model.predict(vec_in)
for i in prediction:
	z = str(i)
	temp = di_i[z]
	result_i.append(temp)
file = open('output_i.txt', 'w')
for item in result_i:
	file.write("%s" % item)
#---------------------------------------------------
#Model predicting just o
#---------------------------------------------------
model = joblib.load('o.pkl')
prediction = model.predict(vec_in)
for i in prediction:
	z = str(i)
	temp = di_o[z]
	result_o.append(temp)
file = open('output_o.txt', 'w')
for item in result_o:
	file.write("%s" % item)
#---------------------------------------------------
#Model predicting just L
#---------------------------------------------------
model = joblib.load('L.pkl')
prediction = model.predict(vec_in)
for i in prediction:
	z = str(i)
	temp = di_L[z]
	result_L.append(temp)
file = open('output_L.txt', 'w')
for item in result_L:
	file.write("%s" % item)
#---------------------------------------------------
#Model predicting just P
#---------------------------------------------------
model = joblib.load('P.pkl')
prediction = model.predict(vec_in)
for i in prediction:
	z = str(i)
	temp = di_P[z]
	result_P.append(temp)
file = open('output_P.txt', 'w')
for item in result_P:
	file.write("%s" % item)
#---------------------------------------------------