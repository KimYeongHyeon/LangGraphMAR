#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <Python.h>
#include <numpy/arrayobject.h>

// C function to perform the calculation
static PyObject* BP(PyObject* self, PyObject* args) {
        
    PyObject* proj;
    PyObject* deg;
    int nview; 
    float dsd; 
    float dso; 
    int nx;
    int ny; 
    float dx; 
    float dy; 
    int nu; 
    float da; 
    float off_a;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "OOiffiiffiff", &proj, &deg, &nview, &dsd, &dso, &nx, &ny, &dx, &dy, &nu, &da, &off_a)) {
        return NULL;  // Return NULL to indicate an error
    }

    // Check if the input objects are sequences (lists or tuples)
    if (!PyList_Check(proj) || !PyList_Check(deg)) {
        PyErr_SetString(PyExc_TypeError, "Input arguments must be lists.");
        return NULL;  // Return NULL to indicate an error
    }

    // Create a Python list to store the result
    int num = nx*ny;
    PyObject* result = PyList_New((num));

    int num_threads = (int) omp_get_max_threads();
    omp_set_num_threads(num_threads);

    float* sino = (float *)calloc(nu*nview, sizeof(float));
    float* img = (float *)calloc(nx*ny, sizeof(float));
    
    for(int jj=0; jj<nu*nview; jj++ ) sino[jj] = (float) PyFloat_AsDouble(PyList_GetItem(proj, jj));

    int ii;
    // #pragma omp parallel for shared(img, sino, nview, dsd, dso, nx, ny, dx, dy, nu, da, off_a) private(ii) reduction(+:img[:nx   *ny])
    #pragma omp parallel for shared(sino, nview, dsd, dso, nx, ny, dx, dy, nu, da, off_a) private(ii) reduction(+:img[:nx*ny])
    for(ii=0; ii<nu*nview; ii++)
    {
        float a, dist, temp, posx, posy, rx, ry, x, y;
        
        int iview = (int) (ii/nu);
        float sinval = (float) sin(PyFloat_AsDouble(PyList_GetItem(deg,iview))/180.0f*3.141592f);
        float cosval = (float) cos(PyFloat_AsDouble(PyList_GetItem(deg,iview))/180.0f*3.141592f);
        
        a = (((ii%nu)  - (nu-1.0f)/2.0f) - off_a)*da ;
        dist = dy/cos(a);
                
        temp = 0.0f;
        
        for(int iy=0; iy<ny; iy++)
        {            
            posy = (iy-(ny-1.0f)/2.0f)*dy;
            posx = tan(a)*(posy+dso);
            
            x = posx*cosval + posy*sinval;
            y = -posx*sinval + posy*cosval;
            
            rx = (x)/dx + (nx-1.0f)/2.0f;
            ry = (y)/dy + (ny-1.0f)/2.0f;
                        
            if(rx>0 && rx<nx-1 && ry>0 && ry<ny-1 )
            {                
                float wx = rx - ((int) rx);               
                float wy = ry - ((int) ry);     
                
                float val = sino[ii]*dist;
                
                img[ ((int) ry)*nx + ((int) rx)] += (1.0f-wx)*(1.0f-wy)*val;
                img[ ((int) ry)*nx + ((int) rx+1)] += (wx)*(1.0f-wy)*val;
                img[ ((int) ry+1)*nx + ((int) rx)] += (1.0f-wx)*(wy)*val;
                img[ ((int) ry+1)*nx + ((int) rx+1)]  += (wx)*(wy)*val;    
              
            }
        }
    }     
        
    for(int jj=0; jj<nx*ny; jj++ ) PyList_SET_ITEM(result, jj, PyFloat_FromDouble((double) img[jj]));
     

    free(img);
    free(sino);
    
    return result;  // Return the result list
}

// C function to perform the calculation
static PyObject* FP(PyObject* self, PyObject* args) {
        
    PyObject* img;
    PyObject* deg;
    int nview; 
    float dsd; 
    float dso; 
    int nx;
    int ny; 
    float dx; 
    float dy; 
    int nu; 
    float da; 
    float off_a;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "OOiffiiffiff", &img, &deg, &nview, &dsd, &dso, &nx, &ny, &dx, &dy, &nu, &da, &off_a)) {
        return NULL;  // Return NULL to indicate an error
    }

    // Check if the input objects are sequences (lists or tuples)
    if (!PyList_Check(img) || !PyList_Check(deg)) {
        PyErr_SetString(PyExc_TypeError, "Input arguments must be lists.");
        return NULL;  // Return NULL to indicate an error
    }

    // Create a Python list to store the result
    int num = nu*nview;
    PyObject* result = PyList_New(num);

    float* sino = (float *)calloc(nu*nview, sizeof(float));
    float* obj = (float *)calloc(nx*ny, sizeof(float));
    
    for(int jj=0; jj<nx*ny; jj++ ) obj[jj] = (float) PyFloat_AsDouble(PyList_GetItem(img, jj));

    omp_set_num_threads(omp_get_max_threads());
    int ii;
    #pragma omp parallel for shared(obj, sino, nview, dsd, dso, nx, ny, dx, dy, nu, da, off_a) private(ii)    
    for(ii=0; ii<nu*nview; ii++)
    {
        float a, dist, temp, posx, posy, rx, ry;
        
        int iview = ((int) ii/nu);
        float sinval = (float) sin(PyFloat_AsDouble(PyList_GetItem(deg,iview))/180.0f*3.141592f);
        float cosval = (float) cos(PyFloat_AsDouble(PyList_GetItem(deg,iview))/180.0f*3.141592f);
        
        a = (((ii%nu)  - (nu-1.0f)/2.0f) - off_a)*da ;
        dist = dy/cos(a);
                
        temp = 0.0f;
        
        for(int iy=0; iy<ny; iy++)
        {            
            posy = (iy-(ny-1.0f)/2.0f)*dy;
            posx = tan(a)*(posy+dso);
            
            rx = (posx*cosval + posy*sinval)/dx + (nx-1.0f)/2.0f;
            ry = (-posx*sinval + posy*cosval)/dy + (ny-1.0f)/2.0f;
                        
            if(rx>0 && rx<nx-1 && ry>0 && ry<ny-1 )
            {                
                float wx = rx - ((int)rx);               
                float wy = ry - ((int)ry);               
                
                temp += ((1.0f-wx)*(1.0f-wy)*obj[ ((int)ry)*nx + ((int)rx)] + (wx)*(1.0f-wy)*obj[ ((int)ry)*nx + ((int)rx+1)] + (1.0f-wx)*(wy)*obj[ ((int)ry+1)*nx + ((int)rx)]+ (wx)*(wy)*obj[ ((int)ry+1)*nx + ((int)rx+1)])*dist;
               
            }
        }
        sino[ii] = temp;
    }   

    for(int jj=0; jj<nu*nview; jj++ ) PyList_SET_ITEM(result, jj, PyFloat_FromDouble((double) sino[jj]));
     
    free(obj);
    free(sino);
   
    return result;  // Return the result list
}

// Method definition
static PyMethodDef methods[] = {
    {"FP", FP, METH_VARARGS, "Perform the forward projection."},
    {"BP", BP, METH_VARARGS, "Perform the backward projection."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "recon",
    NULL,
    -1,
    methods
};

// Module initialization function
PyMODINIT_FUNC PyInit_recon(void) {
    return PyModule_Create(&module);
}




