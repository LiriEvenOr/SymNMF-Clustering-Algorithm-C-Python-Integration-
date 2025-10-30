#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "symnmf.h"

#define MAX_ITERS_NUM (300)
#define BETA (0.5)
#define EPSILON (0.0001)

/* Calculate squared Euclidean distance between two points */
double dist(double *p1, double *p2, int dim){  
    double distance = 0.0;
    int i;
    for(i = 0; i < dim; i++){
        distance += pow(p1[i]- p2[i], 2);
    }
    return distance;
}

/* Initialize a 2D matrix with zeros */
double** initMatrix(int height, int width){
    int i, j;
    double** matrix = (double**)calloc(height, sizeof(double*));
    if(!matrix){
        printf("An Error Has Occurred\n");
        return NULL;
    }
    for(i=0; i<height; i++){
        matrix[i] = (double*)calloc(width, sizeof(double));
        if(!matrix[i]){
            for (j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            printf("An Error Has Occurred\n");
            return NULL;
        }
    }
    return matrix;
}

/* Transpose a matrix */
double** transpose(double** matrix, int height, int width){
    int i, j;
    double** transposed = initMatrix(width, height);
    for(i=0; i<height; i++){
        for(j=0; j<width; j++){
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

/* Free dynamically allocated matrix */
void freeMatrix(double** matrix, int limit){
    int j;
    for (j = 0; j < limit; j++) {
        free(matrix[j]);
    }  
    free(matrix);
}

/* Multiply two matrices */
double** matrixMult(double** A, double** B, int heightA, int widthA, int widthB) {
    int i, j, k;
    double sum = 0;
    double** matrixMult = initMatrix(heightA, widthB);
    for (i=0; i<heightA; i++) {
        for (j=0; j<widthB; j++) {
            for (k=0; k<widthA; k++) {
                sum += (A[i][k]*B[k][j]);
            }
            matrixMult[i][j] = sum;
            sum = 0;
        }
    }
    return matrixMult;
}

/* Subtract matrix mat2 from mat1 */
double** subMatrixes(double** mat1, double** mat2, int height, int width){
    int i, j;
    double** matrixSub = initMatrix(height, width);
    for (i = 0; i < height; i++){
        for(j = 0; j < width; j++){
            matrixSub[i][j] = mat1[i][j] - mat2[i][j];
        }
    }  
    return matrixSub;  
}

/* Calculate squared Frobenius norm */
double squaredForbeniusNorm(double** matrix, int height, int width){
    double sqNorm = 0;
    int i, j;
    for (i = 0; i < height; i++){
        for(j = 0; j < width; j++){
            sqNorm += pow(matrix[i][j], 2);
        }
    }
    return sqNorm;
}

/* Read points from file into matrix and return dimensions */
double** getPointsFromFile(const char* filename, int* numOfPoints, int* dimension){
    double** matrix;
    char* token;
    char buffer[1024];
    int rows = 0, cols = 0, currentRow = 0, currentCol = 0;
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {return NULL;}
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        rows++;
    }
    if (rows == 0) {
        printf("An Error Has Occurred\n");
        exit(1);
    }    
    rewind(fp);
    fgets(buffer, sizeof(buffer), fp);
    token = strtok(buffer, ",");
    while (token != NULL) {
        cols++;
        token = strtok(NULL, ",");
    }
    fclose(fp);
    *numOfPoints = rows;
    *dimension = cols;
    matrix = initMatrix(rows, cols);
    fp = fopen(filename, "r");
    while (fgets(buffer, sizeof(buffer), fp) != NULL && currentRow < rows) {
        currentCol = 0;
        token = strtok(buffer, ",");
        while (token != NULL && currentCol < cols) {
            matrix[currentRow][currentCol] = atof(token);
            currentCol++;
            token = strtok(NULL, ",");
        }
        currentRow++;
    }
    fclose(fp);
    return matrix;
}

/* Print matrix with 4 decimal places */
void printMatrix(double** matrix, int height, int width){
    int i, j;
    for(i = 0; i < height; i++){
        for(j = 0; j < width; j++){
            printf("%.4f", matrix[i][j]);
            if(j != width - 1){
                printf(",");
            }
            else{
                printf("\n");   
            }
        }
    }
}

/* Compute W: symmetric matrix */
double** computeSym(int numOfPoints, int dimension, double** pointsArr){
    int i, j;
    double** symMatrix = initMatrix(numOfPoints, numOfPoints);
    for(i = 0; i < numOfPoints; i++){
        for(j = 0; j < numOfPoints; j++){
            if (i == j) {
                symMatrix[i][j] = 0;
            }
            else {
                symMatrix[i][j] = exp(dist(pointsArr[i], pointsArr[j], dimension) / (-2));
            }
        }
    }
    return symMatrix;
}

/* Compute D: diagonal degree matrix */
double** computeDdg(int numOfPoints, int dimension, double** pointsArr) {
    int i, j;
    double sum = 0;
    double** symMatrix = computeSym(numOfPoints, dimension, pointsArr);
    double** ddgMatrix = initMatrix(numOfPoints, numOfPoints);

    for(i = 0; i < numOfPoints; i++){
        for(j = 0; j < numOfPoints; j++){
            sum += symMatrix[i][j];
        }
        ddgMatrix[i][i] = sum;
        sum = 0;
    }
    freeMatrix(symMatrix, numOfPoints);
    return ddgMatrix;
}

/* Compute normalized matrix: D^(-0.5) * W * D^(-0.5) */
double** computeNorm(int numOfPoints, int dimension, double** pointsArr) {
    int i;
    double** normMatrix, **tempMatrix;
    double** symMatrix = computeSym(numOfPoints, dimension, pointsArr);
    double** ddgMatrix = computeDdg(numOfPoints, dimension, pointsArr);
    for (i=0; i< numOfPoints; i++) {
        ddgMatrix[i][i] = 1/sqrt(ddgMatrix[i][i]);
    }
    tempMatrix = matrixMult(ddgMatrix, symMatrix, numOfPoints, numOfPoints, numOfPoints);
    normMatrix = matrixMult(tempMatrix,ddgMatrix, numOfPoints, numOfPoints, numOfPoints);
    freeMatrix(symMatrix, numOfPoints);
    freeMatrix(ddgMatrix, numOfPoints);
    freeMatrix(tempMatrix, numOfPoints);
    return normMatrix;
}

/* Update H matrix using SymNMF multiplicative rule */
double** updateH(int numOfPoints, int k, double** H, double** W){
    int i, j;
    double** nextH = initMatrix(numOfPoints, k);
    double** numerator = matrixMult(W, H, numOfPoints, numOfPoints, k);
    double** transposed = transpose(H, numOfPoints, k);
    double** tempDenominator = matrixMult(H, transposed, numOfPoints, k, numOfPoints);
    double** denominator = matrixMult(tempDenominator, H, numOfPoints, numOfPoints, k);
    for(i = 0; i<numOfPoints; i++){
        for(j=0; j<k; j++){
            nextH[i][j] = H[i][j]*(1-BETA+BETA*(numerator[i][j]/denominator[i][j]));
        }
    }
    freeMatrix(tempDenominator, numOfPoints);
    freeMatrix(transposed, k);
    freeMatrix(numerator, numOfPoints);
    freeMatrix(denominator, numOfPoints);
    return nextH;
}

double** optimizeH(int numOfPoints, int k, double** H, double** W)
{
    int iterations = 0;
    double** curH = H;
    double** nextH = updateH(numOfPoints, k, curH, W);
    double** differenceMatrix = subMatrixes(nextH, curH, numOfPoints, k);
    while(squaredForbeniusNorm(differenceMatrix, numOfPoints, k) >= EPSILON && iterations < MAX_ITERS_NUM){
        freeMatrix(curH, numOfPoints);
        freeMatrix(differenceMatrix, numOfPoints);
        curH = nextH;
        nextH = updateH(numOfPoints, k, curH, W);
        differenceMatrix = subMatrixes(nextH, curH, numOfPoints, k);
        iterations++;
    }
    freeMatrix(differenceMatrix, numOfPoints);
    freeMatrix(curH, numOfPoints);
    return nextH;
}


int main(int argc, char **argv){
    int numOfPoints, dimension;
    const char* goal = argv[1];
    const char* filename  = argv[2];
    double** matrix;
    double** pointsArr;

    if(argc < 3){
        printf("An Error Has Occurred\n");
        return -1;
    }
    pointsArr = getPointsFromFile(filename, &numOfPoints, &dimension);
    if (pointsArr == NULL) {
        printf("An Error Has Occurred\n");
        return 1;
    }
    if(strcmp(goal, "sym") == 0){
        matrix = computeSym(numOfPoints, dimension, pointsArr);
    } 
    else if (strcmp(goal, "ddg") == 0){
        matrix = computeDdg(numOfPoints, dimension, pointsArr);
    }
    else if (strcmp(goal, "norm") == 0){
        matrix = computeNorm(numOfPoints, dimension, pointsArr);
    }
    else{
        return -1;
    }
    printMatrix(matrix, numOfPoints, numOfPoints);
    freeMatrix(matrix, numOfPoints);
    freeMatrix(pointsArr, numOfPoints);
    return 0;
}