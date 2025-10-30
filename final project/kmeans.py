import math
import sys

EPSILON = 0.0001  # Convergence threshold
ITER = 300  # Default number of iterations
MAX_ITERS_NUM = 1000  # Maximum allowed iterations

# Read the points from the file
def read_points(filename):
    pointsArr = []
    with open(filename, "r") as file:
        for line in file:
            pointsArr.append([float(val) for val in line.strip().split(",")])
    return pointsArr

# Calculate Euclidean distance between two points
def dist(p1, p2):
    dimension = len(p1)
    dist = 0.0
    for i in range(dimension):
        dist += math.pow(p1[i] - p2[i], 2)
    return math.sqrt(dist)

# K-means clustering algorithm
def kmeans(pointsArr, k, iters):
    numOfPoints = len(pointsArr)
    dimension = len(pointsArr[0])
    centroidsArr = []
    for i in range(k):
        centroidsArr.append(pointsArr[i].copy())  # Initialize centroids as first k points

    for it in range(iters):
        newCentroids = [[0.0] * dimension for _ in range(k)]
        countArr = [0] * k
        
        for i in range(numOfPoints):
            closestCentIndex = 0
            minDist = dist(centroidsArr[0], pointsArr[i])
            for index in range(1, k):
                currDist = dist(centroidsArr[index], pointsArr[i])
                if currDist < minDist:
                    closestCentIndex = index
                    minDist = currDist
            countArr[closestCentIndex] += 1
            for d in range(dimension):
                newCentroids[closestCentIndex][d] += pointsArr[i][d]

        for cl in range(k):
            for dim in range(dimension):
                newCentroids[cl][dim] /= countArr[cl]

        smallerThanEps = True
        for c in range(len(centroidsArr)):
            if abs(dist(newCentroids[c], centroidsArr[c])) >= EPSILON:
                smallerThanEps = False

        centroidsArr = newCentroids
        if smallerThanEps:
            break

    return centroidsArr

# Assign labels to points based on closest centroid
def labels(pointsArr, k, iters):
    centroidsArr = kmeans(pointsArr, k, iters)
    labelsArr = []
    for point in pointsArr:
        distances = [dist(point, cent) for cent in centroidsArr]
        labelsArr.append(distances.index(min(distances)))  # Index of nearest centroid
    return labelsArr
