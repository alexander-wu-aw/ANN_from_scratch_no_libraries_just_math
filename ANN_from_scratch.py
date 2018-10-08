# import libraries
import csv
import math
import random

# Import dataset
with open("/Users/alexander/Anaconda Projects/ANN from scratch - no libraries/Churn_Modelling.csv", newline = '') as csvfile:
    dataset = list(csv.reader(csvfile))
# Delete label row
del dataset[0]
# Drop unnecessary columns
for row in range(len(dataset)):
    del dataset[row][0:3]
#Shuffle the dataset
random.shuffle(dataset)


#one hot encoding
def one_hot_encode (dataset, column):
    rows = len(dataset)
    things = []
    for row in range(rows):
        things.append(dataset[row][column])
    items = []
    items.append(dataset[0][column])
    for row in range(rows):
        thing = dataset[row][column]
        if thing not in items:
            items.append(dataset[row][column])
    for item in items:
        for row in range(rows):
            if things[row] == item:
                dataset[row].insert(column, 1)
            else:
                dataset[row].insert(column, 0)
    for row in range(len(dataset)):
        del dataset[row][column+len(items)]
    return dataset

    
country_dataset_ohe = one_hot_encode(dataset = dataset, column = 1)
dataset = one_hot_encode(dataset = country_dataset_ohe, column = 4)

# Making sure all vlaues are doubles
for row in range(len(dataset)):
    dataset[row] = list(map(float, dataset[row]))


# Split dataset into test and train
num_rows = len(dataset)
split = math.floor(num_rows*0.7)
dataset_train = dataset[:split][ :]
dataset_test = dataset[split:][ :]

# Split dataset into X and Y values
num_col = len(dataset[0])
spliting = num_col - 1
dataset_X_train = []
dataset_X_test = []
dataset_Y_train = []
dataset_Y_test = []
for row in range(len(dataset_train)):
    dataset_X_train.append(dataset_train[row][:spliting])
    dataset_Y_train.append(dataset_train[row][spliting:])
for row in range(len(dataset_test)):
    dataset_X_test.append(dataset_test[row][:spliting])
    dataset_Y_test.append(dataset_test[row][spliting:])

# Feature Scaling
def feature_scale_set (dataset):
    mean = []
    for col in range(len(dataset[0])):
        summation = 0
        for row in range(len(dataset)):
            summation = summation + dataset[row][col]
        tempmean = summation / len(dataset)
        mean.append(tempmean)
    maxmin = []
    for col in range(len(dataset[0])):
        largest = dataset[0][col]
        smallest = dataset[0][col]
        for row in range(len(dataset)):
            if dataset[row][col] > largest:
                largest = dataset[row][col]
            if dataset[row][col] < smallest:
                smallest = dataset[row][col]
        tempmaxmin = largest-smallest
        maxmin.append(tempmaxmin)
    return mean, maxmin

def feature_scale (dataset, mean, maxmin):
    num_col = len(dataset[0])
    num_row = len(dataset)
    for col in range(num_col):
        for row in range(num_row):
            dataset[row][col] = (dataset[row][col] - mean[col])/maxmin[col]
    return dataset

mean, maxmin = feature_scale_set(dataset_X_train)
dataset_X_train = feature_scale(dataset_X_train, mean, maxmin)
dataset_X_test = feature_scale(dataset_X_test, mean, maxmin)


# Make each example a column instead of row
def transpose (dataset):
    newdataset = []
    numrows = len(dataset)
    numcols = len(dataset[0])
    for col in range(numcols):
        temprow = []
        for row in range(numrows):
            temprow.append(dataset[row][col])
        newdataset.append(temprow)
    return newdataset

X_train = transpose(dataset_X_train)
X_test = transpose(dataset_X_test)
Y_train = transpose(dataset_Y_train)
Y_test = transpose(dataset_Y_test)
'''
X_train = dataset_X_train
X_test = dataset_X_test
Y_train = dataset_Y_train
Y_test = dataset_Y_test
'''

# The dimensions of each layer
layer_dim = [ len(X_train), 6, 6, 1]

#Initializing parameters
def init_parameters(dim):
    parameters = {}
    num_layers = len(dim)
    for layer in range(1, num_layers):
        matrix = []
        bmatrix = []
        for row in range(dim[layer]):
            temprow = []
            for col in range(dim[layer-1]):
                temprow.append(random.random()*0.1)
            matrix.append(temprow)
            tempzero = [0.0]
            bmatrix.append(tempzero)
        parameters['W' + str(layer)] = matrix
        parameters['b' + str(layer)] = bmatrix
    return parameters

# Calculate the dot product
def dot(matrix1, matrix2):
    matrix = []
    col2 = len(matrix2[0])
    row1 = len(matrix1)
    dims = len(matrix2)
    if dims == len(matrix1[0]):
        for row in range(row1):
            temprow = []
            for col in range(col2):
                value = 0
                for dim in range(dims):
                    value = value + (matrix1[row][dim] * matrix2[dim][col])
                temprow.append(value)
            matrix.append(temprow)
        return matrix
    else:
        print("The column of first matrix doesn't match with row of second matrix")

#add the bias matrix element-wise
def addb(matrix, b):
    newmatrix = []
    numrows = len(b)
    numcols = len(matrix[0])
    if numrows == len(b):
        for row in range(numrows):
            temprow = []
            for col in range(numcols):
                temprow.append(matrix[row][col] + b[row][0])
            newmatrix.append(temprow)
        return newmatrix
    else:
        print("The dimensions don't match. The column of each column doesn't match")
        
### this is to do sigmoid activation
def sigmoid(matrix):
    newmatrix = []
    numrows = len(matrix)
    numcols = len(matrix[0])
    for row in range(numrows):
        temprow = []
        for col in range(numcols):
            value = matrix[row][col] * -1
            value2 = math.exp(value)
            value3 = 1 + value2
            value4 = 1/value3
            temprow.append(value4)
        newmatrix.append(temprow)
    return newmatrix        

# Multiply element-wise
def multiply(matrix, num):
    newmatrix = []
    numrows = len(matrix)
    numcols = len(matrix[0])
    for row in range(numrows):
        temprow = []
        for col in range(numcols):
            temprow.append(matrix[row][col] * num)
        newmatrix.append(temprow)
    return newmatrix

# Calculate exponents for each element
def exp (matrix):
    newmatrix = []
    rows = len(matrix)
    cols = len(matrix[0])
    for row in range(rows):
        temprow=[]
        for col in range(cols):
            temprow.append(math.exp(matrix[row][col]))
        newmatrix.append(temprow)
    return newmatrix



# Forward propogation up until before output layer
def forward(X, parameters, dim):
    cache = {}
    cache['A0'] = X
    for layer in range(1, len(dim)-1):
        Zpre = dot(parameters['W' + str(layer)], cache['A' + str(layer-1)])
        Z = addb(Zpre, parameters['b' + str(layer)] )
        cache['Z' + str(layer)] = Z
        A = []
        for row in range(len(Z)):
            temprow = []
            for col in range(len(Z[0])):
                if Z[row][col] < 0:
                    temprow.append(0)
                else:
                    temprow.append(Z[row][col])
            A.append(temprow)
        cache['A' + str(layer)] = A
    return cache

#Now do the output layer
def output(cache, parameters, dim):
    layer = len(dim) -1
    Zpre = dot(parameters['W' + str(layer)], cache['A' + str(layer-1)])
    Z = addb(Zpre, parameters['b' + str(layer)] )
    cache['Z' + str(layer)] = Z
    A = sigmoid(Z)
    cache['A' + str(layer)] = A
    return cache

# Cost function
def cost_function(Y, cache, dim):
    output = cache['A' + str(len(dim)-1)]
    m = len(Y)
    if m == len(output):
        loss = 0
        for example in range(m):
            loss = loss + ((Y[0][example] * math.log(output[0][example])) + ((1-Y[0][example]) * math.log(1-output[0][example])))
        cost = (loss/m) * - 1
        return cost
    else:
        print("The number of examples for the ouput and the answer doesn't match")
        
# Gradient for output
def get_gradients(Y, parameters, cache, dim):
    gradients = {}
    # Find final layer dZ
    temprow = []
    dzfinal = []
    for example in range(len(Y[0])):
        temprow.append(cache['A' + str(len(dim) -1)][0][example] - Y[0][example])
    dzfinal.append(temprow)
    gradients['dZ' + str(len(dim) -1)] = dzfinal
    # Find final layer dW
    gradients['dW' + str(len(dim) -1)] = multiply(dot(gradients['dZ'+ str(len(dim) -1)], transpose(cache['A' + str(len(dim) -2)])),(1/len(Y)))

    # Find final layer dB
    db = []
    dbpre= []
    bsum = 0
    for example in range(len(Y[0])):
        bsum = bsum + gradients['dZ' + str(len(dim) -1)][0][example]
    dbpre.append(bsum/len(Y[0]))
    db.append(dbpre)
    gradients['db' + str(len(dim) -1)] = db
    # For all subsequent layers
    for layer in range(len(dim)-2,0,-1):
        dZpre = dot(transpose(parameters['W' + str(layer+1)]),  gradients['dZ' + str(layer+1)])
        dZ = []
        for row in range(len(dZpre)):
            temprow = []
            for col in range(len(dZpre[0])):
                value = cache['Z' + str(layer)][row][col]
                if value < 0:
                    temprow.append(0)
                else:
                    temprow.append(value)
            dZ.append(temprow)
        gradients['dZ' + str(layer)] = dZ
        
        gradients['dW' + str(layer)] = multiply(dot(gradients['dZ'+ str(layer)], transpose(cache['A' + str(layer-1)])),(1/len(Y)))
        
        db = []
        for row in range(len(gradients['dZ' + str(layer)])):
            rowsum = 0
            bsum = []
            for col in range(len(gradients['dZ' + str(layer)][0])):
                rowsum = rowsum + gradients['dZ' + str(layer)][row][col]
            bsum.append(rowsum/len(Y[0]))
            db.append(bsum)
        gradients['db' + str(layer)] = db

    return gradients

# Element wise subtraction
def subtract(matrix1, matrix2):
    newmatrix= []
    numrows = len(matrix1)
    numcols = len(matrix1[0])
    if numrows == len(matrix2) and numcols == len(matrix2[0]):
        for row in range(numrows):
            temprow = []
            for col in range(numcols):
                value = matrix1[row][col] - matrix2[row][col]
                temprow.append(value)
            newmatrix.append(temprow)
        return newmatrix
    else:
        "The dimensions of your matrices do not match"

# Gradient descent
def gradient_descent(parameters, gradients, dim):
    for layer in range(1, len(dim)):
        parameters['W' + str(layer)] = subtract(parameters['W' + str(layer)], gradients['dW' + str(layer)])
        parameters['b' + str(layer)] = subtract(parameters['b' + str(layer)], gradients['db' + str(layer)])
    return parameters

parameters = init_parameters(layer_dim)
for x in range(50):
    print('Gradient descent step number: ' + str(x))
    cache = forward(X_train, parameters, layer_dim)
    cache = output(cache, parameters, layer_dim)
    cost = cost_function(Y_train, cache, layer_dim)
    print(cost)
    gradients = get_gradients(Y_train, parameters, cache, layer_dim)
    parameters = gradient_descent(parameters, gradients, layer_dim)
    
answer = Y_train[0]
total = len(answer)
guess = []
preguess = cache['A' + str(len(layer_dim)-1)][0]
numcols = len(cache['A' + str(len(layer_dim)-1)][0])
for col in range(numcols):
    if preguess [col] < 0.5:
        guess.append(0)
    else:
        guess.append(1)
right = 0
wrong = 0
for example in range(total):
    if answer[example] == guess[example]:
        right = right + 1
    else:
        wrong = wrong + 1
print()
print("You got " + str( 100 * (right/total)) + "% right on the training set")
print("Your final error is " + str( cost))
print()
    
def result(X_test, Y_test, parameters, layer_dim):
    cachetest = forward(X_test, parameters, layer_dim)
    cachetest = output(cachetest, parameters, layer_dim)
    cost = cost_function(Y_test, cachetest, layer_dim)
    total = len(Y_test[0])
    answer = Y_test[0]
    preguess = cachetest['A' + str(len(layer_dim)-1)][0]

    guess = []
    numcols = len(cachetest['A' + str(len(layer_dim)-1)][0])
    for col in range(numcols):
        if preguess [col] < 0.5:
            guess.append(0)
        else:
            guess.append(1)
    right = 0
    wrong = 0
    for example in range(total):
        if answer[example] == guess[example]:
            right = right + 1
        else:
            wrong = wrong + 1
    print("You got " + str( 100 * (right/total)) + "% right on the test set")
    print("Your test error is " + str( cost))
        
result(X_test, Y_test, parameters, layer_dim)
        
        
        