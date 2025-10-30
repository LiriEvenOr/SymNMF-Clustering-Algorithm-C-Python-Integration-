import sys
import kmeans as km
import symnmf as sn
from sklearn.metrics import silhouette_score

DEFAULT_ITERS = 200  # Default number of iterations for k-means

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
        return matrix  # Return list of points as lists of floats
    except:
        print("An Error Has Occurred")
        exit()  # Exit on file reading error

try:
    k = int(sys.argv[1])  # Get number of clusters
except:
    print("An Error Has Occurred")
    exit()

filename = sys.argv[2]  # Get input filename
points = getPyPointsFromFile(filename)  # Read data points from file

# Validate k
if(k >= len(points) or k < 2):
    print("An Error Has Occurred")
    exit()

# Get clustering labels from k-means and SymNMF
kmeansLabels = km.labels(points, k, DEFAULT_ITERS)
symnmfLabels = sn.labels(points, k)

# Calculate silhouette scores for each clustering
kmeansScore = silhouette_score(points, kmeansLabels)
symnmfScore = silhouette_score(points, symnmfLabels)

# Print scores
print("nmf: " + "%.4f"%symnmfScore)
print("kmeans: " + "%.4f"%kmeansScore)
