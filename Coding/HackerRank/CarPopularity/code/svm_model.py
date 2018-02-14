import pandas as pd
from sklearn import svm
import numpy as np
from eval import print_f1_score

df = pd.read_csv("./data/train.csv")
clf = svm.SVC(gamma=0.001, C=100., decision_function_shape='ovo')
clf.fit( df.iloc[:, :-1], df.popularity[:] )
pred = clf.predict( df.iloc[:, :-1] )

np.count_nonzero( pred==df.popularity )

test_df = pd.read_csv("./data/test.csv")

print_f1_score( df.iloc[:,-1].values, pred );
