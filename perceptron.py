import numpy as np
# Setting the random seed, it can be changed to see different solutions.
np.random.seed(40)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])


def perceptronStep(X, y, W, b, learn_rate = 0.01):
    """
    Update weights and bias whenever needed.
    
    Parameters:
    ----------
        X: a n_by_2 numpy 2D array. Each entry represents a set of coordinates. 
           It represents the 0th and 1st column in data.csv.
           
        y: actual labels for the datapoints. It represents the 2nd column in data.csv
        
        W: a 2_by_1 2D numpy array. Weights for the linear equation.
        
        b: Bias value for the linear equation.
        
        learning_rate: float. small steps in changing weights and bias when necessary.
     
     Returns:
        new weights and bias.    
    
    """
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        
        if y[i]-y_hat == 1:  # predicted neg but actually it's pos
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
            
        elif y[i]-y_hat == -1:  # predicted pos but actually it's neg
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
            
    return W, b    


def trainPerceptronAlgorithm(X, y, learn_rate = 0.001, num_epochs = 50):
    """
    Runs the perceptron algorithm repeatedly on the dataset, and returns 
    a few of the boundary lines obtained in the iterations, for plotting purposes.
    """
    x_min, x_max = min(X.T[0]), max(X.T[0])  # the minimum and maximun value in Variable x0 (column 0 in data.csv)
    y_min, y_max = min(X.T[1]), max(X.T[1])  # the minimum and maximun value in Variable x1 (column 1 in data.csv)
    
    W = np.array(np.random.rand(2,1))  # Initialize the W weights between (0, 1)
    b = np.random.rand(1)[0] + x_max  # Initialize the Bias b between (0, 1)
    
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        
        # each entry in the boundary_lines represents a linear equation.
        boundary_lines.append((-W[0]/W[1], -b/W[1])) 
    print(boundary_lines)
    return boundary_lines
