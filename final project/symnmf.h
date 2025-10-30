#ifndef SYMNMF_H
#define SYMNMF_H

void freeMatrix(double** matrix, int limit);
double** computeSym(int numOfPoints, int dimension, double** pointsArr);
double** computeDdg(int numOfPoints, int dimension, double** pointsArr);
double** computeNorm(int numOfPoints, int dimension, double** pointsArr);
double** optimizeH(int numOfPoints, int dimension, double** H, double** W);

#endif


