import sys
import math
import numpy as np
import mysymnmf as sn

np.random.seed(1234)

# Calculates the average value of all entries in a square matrix
def averageEntry(matrix, size):
    sum = 0
    for i in range(size):
        for j in range(size):
            sum += matrix[i][j]
    average = sum / (size * size)
    return average

# Initializes matrix H with random values between 0 and a calculated boundary
def initH(W, n, k):
    boundary = 2 * math.sqrt(averageEntry(W, n) / k)
    H = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            H[i][j] = np.random.uniform(low=0, high=boundary)
    return H

# Reads a CSV file and returns a list of point vectors as floats
def getPyPointsFromFile(filename):
    matrix = []
    try:
        with open(filename, "r") as f:
            first_line = f.readline().strip()      
            tokens = first_line.split(",")
            row = [float(token) for token in tokens]
            matrix.append(row)

            for line in f:
                line = line.strip()
                tokens = line.split(",")
                row = [float(token) for token in tokens]
                matrix.append(row)
        return matrix
    except:
        # Catches all exceptions and prints generic error message
        print("An Error Has Occurred")
        exit()

# Prints matrix with 4 decimal precision, comma-separated format
def printMatrix(matrix):
    height = len(matrix)
    width = len(matrix[0])
    for i in range(height):
        for j in range(width - 1):
            formatted = "%.4f"%matrix[i][j]
            print(f"{formatted},", end="")
        formatted = "%.4f"%matrix[i][width - 1]
        print(formatted)

# Computes the final H matrix using symnmf with normalized similarity matrix
def computeH(points, k):
    numOfPoints = len(points)
    W = sn.norm(points)
    H = initH(W, numOfPoints, k)
    H = H.tolist()
    out = sn.symnmf(W, H, k)
    return out

# Assigns each point a label based on the index of the highest value in its H row
def labels(points, k):
    H = computeH(points, k)
    labels = []
    for i in range(len(points)):
        max = H[i][0]
        maxIdx = 0
        for j in range(1, k):
            if(H[i][j] > max):
                max = H[i][j]
                maxIdx = j
        labels.append(maxIdx) 
    return labels


try:
    k = int(sys.argv[1])  # Parses number of clusters k from command line
except:
    print("An Error Has Occurred")
    exit()
goal = sys.argv[2]  # Parses goal: sym, ddg, norm, or symnmf

if(len(sys.argv) == 4):
    filename = sys.argv[3]
    pointsArr = getPyPointsFromFile(filename)

    # Validates that k is within bounds
    if(k >= len(pointsArr) or k < 0):
        print("An Error Has Occurred")
        exit()

    if(goal == "sym"):
        out = sn.sym(pointsArr)
        printMatrix(out)

    elif(goal == "ddg"):
        out = sn.ddg(pointsArr)
        printMatrix(out)

    elif(goal == "norm"):
        out = sn.norm(pointsArr)
        printMatrix(out)

    elif(goal == "symnmf"):
        if(k < 2):  # symnmf only valid for k >= 2
            print("An Error Has Occurred")
            exit()
        out = computeH(pointsArr, k)
        printMatrix(out)

elif(len(sys.argv) < 3): 
    # Handles case of insufficient arguments
    print("An Error Has Occurred")
    exit()
