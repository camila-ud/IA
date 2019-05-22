from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


### This files reloads the pyrat_dataset that was stored as a pkl file by the generate dataset script. 
import numpy as np
import random as rd
import pickle
from sklearn.externals import joblib
mazeWidth = 21
mazeHeight = 15

import pickle, scipy


x,y = pickle.load(open("pyrat_dataset.pkl","rb"))

## As the dataset was stored using scipy sparse array to save space, we convert it back to numpy dense array. 
## Note that you could keep the sparse representation if you work with a machine learning method that accepts sparse arrays. 
x = scipy.sparse.vstack(x).todense()
y = scipy.sparse.vstack(y).todense()

x = np.array(x).reshape(-1,(2*mazeHeight-1)*(2*mazeWidth-1))
y = np.argmax(np.array(y),1)

### Now you have to train a classifier using supervised learning and evaluate it's performance. 

## To be completed
print(x.shape,y.shape)



### Let's assume you have named your classifier clf . You can save the trained object using the joblib.dump method, as follows: 

#joblib.dump(clf, 'save.pkl') 

# Test in pyrat
## Now you can use the supervised.py file as an AI directly in Pyrat. 
# copy save.pkl, utils.py to the pyrat root folder and supervised.py to the pyrat root/AIs folder
