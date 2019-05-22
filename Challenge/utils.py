import numpy as np

### The goal of this function is to create a canvas, which will be the vector used to train the classifier. 
### As we want to predict a next move, we will create a canvas that is centered on the player, so that we can easily with the translation invariance. 


### The goal of this function is to create a canvas, which will be the vector used to train the classifier. 
### As we want to predict a next move, we will create a canvas that is centered on the player, so that we can easily with the translation invariance. 


def convert_input_2(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
	# We will consider twice the size of the maze to simplify the creation of the canvas 
	# The canvas is initialized as a numpy tensor with 3 modes (meaning it is indexed using three integers), the third one corresponding to "layers" of the canvas. 
	# Here, we just use one layer, but you can defined other ones to put more information on the play (e.g. the location of the opponent could be put in a second layer)

    im_size = (2*mazeHeight-1,2*mazeWidth-1,1)

    # We initialize a canvas with only zeros
    canvas = np.zeros(im_size)


    (x,y) = player
    (x_op,y_op) = opponent

    # fill in the first layer of the canvas with the value 1 at the location of the cheeses, relative to the position of the player (i.e. the canvas is centered on the player location)
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese,y_cheese) in piecesOfCheese:
        canvas[y_cheese+center_y-y,x_cheese+center_x-x,0] = 1
    #add position opp
    canvas[y_op+center_y-y,x_op+center_x-x,0] = canvas[y_op+center_y-y,x_op+center_x-x] + 2
    return canvas
