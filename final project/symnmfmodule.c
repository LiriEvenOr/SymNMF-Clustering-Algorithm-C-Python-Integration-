#define PY_SSIZE_T_CLEAN
# include <Python.h>
# include "symnmf.h"

/* Converts a Python matrix to a C matrix (2D array) */
double** makeCMatrix(int height, int width, PyObject* pyMatrix){
    int i, j;
    PyObject *line, *item;
    double** matrix = (double **)malloc(height * sizeof(double *));
    if(!matrix){
        printf("An Error Has Occurred\n");
        return NULL;
    }
    for(i=0; i<height; i++)
    {
        matrix[i] = (double *)malloc(width * sizeof(double));
        if(!matrix[i]){
            for (j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            printf("An Error Has Occurred\n");
            return NULL;
        }
        line = PyList_GetItem(pyMatrix, i);  /* Get each line of the Python list */
        for(j=0; j<width; j++)
        {
            item = PyList_GetItem(line, j);  /* Get each item in the line */
            matrix[i][j] = PyFloat_AsDouble(item);  /* Convert Python float to C double */
        }
    }
    return matrix;  /* Return the converted C matrix */
}

/* Converts a C matrix (2D array) to a Python matrix (list of lists) */
static PyObject* makePyMatrix(int height, int width, double **cMatrix){
    int i, j;
    PyObject *pyMatrix, *line;
    
    pyMatrix = PyList_New(height);  /* Create a new Python list for the matrix */
    for (i = 0; i < height; i++)
    {
        line = PyList_New(width);  /* Create a new Python list for each row */
        for (j = 0; j < width; j++)
        {
            PyList_SetItem(line, j, PyFloat_FromDouble(cMatrix[i][j]));  /* Convert each C double to a Python float */
        }
        PyList_SetItem(pyMatrix, i, line);  /* Set the row in the matrix */
    }
    return pyMatrix;  /* Return the converted Python matrix */
}

/* Function to compute the final H matrix using symnmf */
static PyObject* symnmf(PyObject* self, PyObject* args){
    PyObject *W, *H, *pyFinalH;
    double **cWMatrix, **cHMatrix, **cFinalH;
    int numOfPoints, k;

    if(!PyArg_ParseTuple(args, "OOi", &W, &H, &k)){  /* Parse arguments */
        printf("An Error Has Occurred\n");
        return NULL;
    }
    numOfPoints = PyList_Size(H);  /* Get the size of H matrix */
    cWMatrix = makeCMatrix(numOfPoints, numOfPoints, W);  /* Convert W to C matrix */
    cHMatrix = makeCMatrix(numOfPoints, k, H);  /* Convert H to C matrix */
    cFinalH = optimizeH(numOfPoints, k, cHMatrix, cWMatrix);  /* Optimize H matrix */
    pyFinalH = makePyMatrix(numOfPoints, k, cFinalH);  /* Convert the final C matrix back to Python */
    freeMatrix(cWMatrix, numOfPoints);  /* Free allocated memory for W matrix */
    freeMatrix(cFinalH, numOfPoints);  /* Free allocated memory for final H matrix */
    return pyFinalH;  /* Return the final Python matrix */
}

/* Function to compute the similarity matrix (sym) */
static PyObject* sym(PyObject* self, PyObject* args){
    PyObject *pointsArr, *line, *pySymMatrix;
    double **cPointsArr, **cSymMatrix;
    int numOfPoints, dimension;

    if(!PyArg_ParseTuple(args, "O", &pointsArr)){  /* Parse the points array argument */
        printf("An Error Has Occurred\n");
        return NULL;
    }
    numOfPoints = PyList_Size(pointsArr);  /* Get the number of points */
    line = PyList_GetItem(pointsArr, 0);  /* Get the first point to determine the dimension */
    dimension = PyList_Size(line);  /* Get the dimension of the points */
    cPointsArr = makeCMatrix(numOfPoints, dimension, pointsArr);  /* Convert points array to C matrix */
    cSymMatrix = computeSym(numOfPoints, dimension, cPointsArr);  /* Compute the similarity matrix */
    pySymMatrix = makePyMatrix(numOfPoints, numOfPoints, cSymMatrix);  /* Convert the C matrix back to Python */
    freeMatrix(cPointsArr, numOfPoints);  /* Free allocated memory for points matrix */
    freeMatrix(cSymMatrix, numOfPoints);  /* Free allocated memory for similarity matrix */
    return pySymMatrix;  /* Return the Python similarity matrix */
}

/* Function to compute the diagonal degree matrix (ddg) */
static PyObject* ddg(PyObject* self, PyObject* args){
    PyObject *pointsArr, *line, *pyDdgMatrix;
    double **cPointsArr, **cDdgMatrix;
    int numOfPoints, dimension;

    if(!PyArg_ParseTuple(args, "O", &pointsArr)){  /* Parse the points array argument */
        printf("An Error Has Occurred\n");
        return NULL;
    }
    numOfPoints = PyList_Size(pointsArr); 
    line = PyList_GetItem(pointsArr, 0);
    dimension = PyList_Size(line);
    cPointsArr = makeCMatrix(numOfPoints, dimension, pointsArr);  /* Convert points array to C matrix */
    cDdgMatrix = computeDdg(numOfPoints, dimension, cPointsArr);  /* Compute the diagonal degree matrix */
    pyDdgMatrix = makePyMatrix(numOfPoints, numOfPoints, cDdgMatrix);  /* Convert the C matrix back to Python */
    freeMatrix(cPointsArr, numOfPoints); 
    freeMatrix(cDdgMatrix, numOfPoints);
    return pyDdgMatrix;  /* Return the Python diagonal degree matrix */
}

/* Function to compute the normalized similarity matrix (norm) */
static PyObject* norm(PyObject* self, PyObject* args){
    PyObject *pointsArr, *line, *pyNormMatrix;
    double **cPointsArr, **cNormMatrix;
    int numOfPoints, dimension;

    if(!PyArg_ParseTuple(args, "O", &pointsArr)){  /* Parse the points array argument */
        printf("An Error Has Occurred\n");
        return NULL;
    }
    numOfPoints = PyList_Size(pointsArr);
    line = PyList_GetItem(pointsArr, 0);
    dimension = PyList_Size(line); 
    cPointsArr = makeCMatrix(numOfPoints, dimension, pointsArr);  /* Convert points array to C matrix */
    cNormMatrix = computeNorm(numOfPoints, dimension, cPointsArr);  /* Compute the normalized similarity matrix */
    pyNormMatrix = makePyMatrix(numOfPoints, numOfPoints, cNormMatrix);  /* Convert the C matrix back to Python */
    freeMatrix(cPointsArr, numOfPoints);  
    freeMatrix(cNormMatrix, numOfPoints);
    return pyNormMatrix;  /* Return the Python normalized similarity matrix */
}

/* Method definitions for the module */
static PyMethodDef symnmfMethods [] = {
    {"symnmf", (PyCFunction) symnmf, METH_VARARGS, PyDoc_STR("compute final H") },
    {"sym", (PyCFunction) sym, METH_VARARGS, PyDoc_STR("compute similarity matrix") },
    {"ddg", (PyCFunction) ddg, METH_VARARGS, PyDoc_STR("compute diagonal degree matrix") },
    {"norm", (PyCFunction) norm, METH_VARARGS, PyDoc_STR("compute normalized similarity matrix") },
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf",
    NULL,
    -1,  
    symnmfMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_mysymnmf(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule); 
    if (!m) {
        return NULL;
    }
    return m;
}
