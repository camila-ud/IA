import scipy
import scipy.sparse
import utils

# The tqdm package is useful to visualize progress with long computations. 
# Install it using pip 
import tqdm

import numpy as np
import ast
import os

PHRASES = {
    "# Random seed\n": "seed",
    "# MazeMap\n": "maze",
    "# Pieces of cheese\n": "pieces"    ,
    "# Rat initial location\n": "rat"    ,
    "# Python initial location\n": "python"   , 
    "rat_location then python_location then pieces_of_cheese then rat_decision then python_decision\n": "play"
}

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

translate_action = {
    MOVE_LEFT:0,
    MOVE_RIGHT:1,
    MOVE_UP:2,
    MOVE_DOWN:3
}# This data structures defines the encoding of the four possible movements

def process_file_2(filename):
    f = open(filename,"r")    
    info = f.readline()
    params = dict(play=list())
    while info is not None:
        if info.startswith("{"):
            params["end"] = ast.literal_eval(info)
            break
        if "turn " in info:
            info = info[info.find('rat_location'):]
        if info in PHRASES.keys():
            param = PHRASES[info]
            if param == "play":
                rat = ast.literal_eval(f.readline())
                python = ast.literal_eval(f.readline())
                pieces = ast.literal_eval(f.readline())
                rat_decision = f.readline().replace("\n","")
                python_decision = f.readline().replace("\n","")
                play_dict = dict(
                    rat=rat,python=python,piecesOfCheese=pieces,
                    rat_decision=rat_decision,python_decision=python_decision)
                params[param].append(play_dict)
            else:
                params[param] = ast.literal_eval(f.readline())
        else:
            print("did not understand:", info)
            break
        info = f.readline()
    return params

def dict_to_x_y(end,rat, python, maze, piecesOfCheese,rat_decision,python_decision,
                mazeWidth=21, mazeHeight=15):
    # We only use the winner
    if end["win_python"] == 1: 
        player = python
        opponent = rat        
        decision = python_decision
    elif end["win_rat"] == 1:
        player = rat
        opponent = python        
        decision = rat_decision
    else:
        return False
    if decision == "None" or decision == "": #No play
        return False
    x_1 = utils.convert_input_2(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese)
    y = np.zeros((1,4),dtype=np.int8)
    y[0][translate_action[decision]] = 1
    return x_1,y

games = list()
directory = "/home/brain/IA/PyRat/saves/"
for root, dirs, files in os.walk(directory):
    for filename in tqdm.tqdm(files):
        if filename.startswith("."):
            continue
        game_params = process_file_2(directory+filename)
        games.append(game_params)

x_1_train = list()
y_train = list()
wins_python = 0
wins_rat = 0
for game in tqdm.tqdm(games):
    if game["end"]["win_python"] == 1: 
        wins_python += 1
    elif game["end"]["win_rat"] == 1:
        wins_rat += 1
    else:
        continue
    plays = game["play"]
    for play in plays:
        x_y = dict_to_x_y(**play,maze=game_params["maze"],end=game["end"])
        if x_y:
            x1, y = x_y
            y_train.append(scipy.sparse.csr_matrix(y.reshape(1,-1)))
            x_1_train.append(scipy.sparse.csr_matrix(x1.reshape(1,-1)))
print("Greedy/Draw/Greedy, {}/{}/{}".format(wins_rat,1000 - wins_python - wins_rat, wins_python)) 

# dataset moves
np.savez_compressed("dataset_challenge_moves_supervised_opponent_position.npz",x=x_1_train,y=y_train)
del x_1_train
del y_train

from sklearn.model_selection import train_test_split

### This cell reloads the pyrat_dataset that was stored as a pkl file by the generate dataset script. 

mazeWidth = 21
mazeHeight = 15

import pickle, scipy

x = np.load("dataset_challenge_moves_supervised_opponent_position.npz")['x']
y = np.load("dataset_challenge_moves_supervised_opponent_position.npz")['y']

x = scipy.sparse.vstack(x)

## The dataset was stored using scipy sparse arrays, because the matrices contain mostly zeros. In case you wish to use 
## supervised learning techniques that don't accept sparse matrices, you have to convert x into a dense array and reshape it accordingly
#x = x.todense()
#x = np.array(x).reshape(-1,(2*mazeHeight-1)*(2*mazeWidth-1))

y = scipy.sparse.vstack(y).todense()
y = np.argmax(np.array(y),1)

print("Dataset load: ",x.shape,y.shape) 

from sklearn.neural_network import MLPClassifier

### Now you have to train a classifier using supervised learning and evaluate it's performance. 
#Split dataset

x_train, x_test, y_train, y_test = train_test_split(x[:,:], y[:], test_size=0.20, random_state=1)

clf = MLPClassifier(verbose = 1)
clf.fit(x_train,y_train)
print(clf.score(x_train,y_train),clf.score(x_test,y_test))

from sklearn.metrics import classification_report,confusion_matrix
y_pred_train = clf.predict(x_train)
report = classification_report(y_true=y_train,y_pred=y_pred_train)

print("Train Set:")
print(report)


y_pred_test = clf.predict(x_test)
report = classification_report(y_true=y_test,y_pred=y_pred_test)

print("Test Set:")
print(report)

### Let's assume you have named your classifier clf . You can save the trained object using the joblib.dump method, as follows: 

import pickle
from sklearn.externals import joblib

joblib.dump(clf, 'mlp_classifier_moves_with_opponent_pos.pkl') 
print("model saved")
# Test in pyrat
## Now you can use the supervised.py file as an AI directly in Pyrat. 
