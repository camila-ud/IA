import numpy as np

### The goal of this function is to create a canvas, which will be the vector used to train the classifier. 
### As we want to predict a next move, we will create a canvas that is centered on the player, so that we can easily with the translation invariance. 


def convert_input(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
	# We will consider twice the size of the maze to simplify the creation of the canvas 
	# The canvas is initialized as a numpy tensor with 3 dimensions, the third one corresponding to "layers" of the canvas. 
	# Here, we just use one layer, but you can defined other ones to put more information on the play (e.g. the location of the opponent could be put in a second layer)

    im_size = (2*mazeHeight-1,2*mazeWidth-1,1)

    # We initialize a canvas with only zeros
    canvas = np.zeros(im_size)
    print(canvas.shape)


    (x,y) = player
    # To be completed : fill in the first layer of the canvas with the value 1 at the location of the cheeses, relative to the position of the player (i.e. the canvas is centered on the player location)
    for (x_cheese,y_cheese) in piecesOfCheese:
        canvas[x_cheese-x,y_cheese-y,0] = 1 
	#centering position
    return canvas
