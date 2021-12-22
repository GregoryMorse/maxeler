#ifndef ZOnePermanentCalculator_wrapper_H
#define ZOnePermanentCalculator_wrapper_H


#include <Python.h>
#include <numpy/arrayobject.h>
#include "structmember.h"
#include "ZOnePermanentCalculator.h"
#include "ZOnePermanentCalculatorDFE.h"
#include "numpy_interface.h"



/**
@brief Type definition of the ZOnePermanentCalculator_wrapper Python class of the ZOnePermanentCalculator_wrapper module
*/
typedef struct ZOnePermanentCalculator_wrapper {
    PyObject_HEAD
    /// The C++ variant of class CZOnePermanentCalculator
    pic::ZOnePermanentCalculator* calculator;
} ZOnePermanentCalculator_wrapper;


/**
@brief Creates an instance of class ZOnePermanentCalculator and return with a pointer pointing to the class instance (C++ linking is needed)
@return Return with a void pointer pointing to an instance of N_Qubit_Decomposition class.
*/
pic::ZOnePermanentCalculator*
create_ZOnePermanentCalculator() {

    return new pic::ZOnePermanentCalculator();
}

/**
@brief Call to deallocate an instance of ZOnePermanentCalculator class
@param ptr A pointer pointing to an instance of ZOnePermanentCalculator class.
*/
void
release_ZOnePermanentCalculator( pic::ZOnePermanentCalculator*  instance ) {
    if ( instance != NULL ) {
        delete instance;
    }
    return;
}





extern "C"
{


/**
@brief Method called when a python instance of the class ZOnePermanentCalculator_wrapper is destroyed
@param self A pointer pointing to an instance of class ZOnePermanentCalculator_wrapper.
*/
static void
ZOnePermanentCalculator_wrapper_dealloc(ZOnePermanentCalculator_wrapper *self)
{

    // deallocate the instance of class N_Qubit_Decomposition
    release_ZOnePermanentCalculator( self->calculator );
   
    Py_TYPE(self)->tp_free((PyObject *) self);

    // unload DFE
    //releive_ZOne_DFE();
}

/**
@brief Method called when a python instance of the class ZOnePermanentCalculator_wrapper is allocated
@param type A pointer pointing to a structure describing the type of the class ZOnePermanentCalculator_wrapper.
*/
static PyObject *
ZOnePermanentCalculator_wrapper_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    ZOnePermanentCalculator_wrapper *self;
    self = (ZOnePermanentCalculator_wrapper *) type->tp_alloc(type, 0);
    if (self != NULL) {}


    return (PyObject *) self;
}


/**
@brief Method called when a python instance of the class ZOnePermanentCalculator_wrapper is initialized
@param self A pointer pointing to an instance of the class ZOnePermanentCalculator_wrapper.
@param args A tuple of the input arguments: qbit_num (integer)
qbit_num: the number of qubits spanning the operations
@param kwds A tuple of keywords
*/
static int
ZOnePermanentCalculator_wrapper_init(ZOnePermanentCalculator_wrapper *self)
{

    // initialize DFE array
    //initialize_ZOne_DFE();

    // create instance of class ZOnePermanentCalculator
    self->calculator = create_ZOnePermanentCalculator();

    return 0;
}

/**
@brief Call to create a PIC matrix representation of a numpy array
*/
std::vector<uint64_t>
numpybits2matrix(PyObject *arr) {

    if ( arr == Py_None ) {
        return std::vector<uint64_t>();
    }

#ifdef DEBUG
    // test C-style contiguous memory allocation of the arrays
    // in production this case has to be handled outside
    assert( PyArray_IS_C_CONTIGUOUS(arr) && "array is not memory contiguous" );
#endif

    // get the pointer to the data stored in the input matrices
    // must be uint_8 data
    uint8_t* data = (uint8_t*)PyArray_DATA(arr);

    // get the dimensions of the array self->C
    int dim_num = PyArray_NDIM( arr );
    npy_intp* dims = PyArray_DIMS(arr);
    
    // create PIC version of the input matrices
    if (dim_num == 2) {
        assert(dims[0] <= 64 && dims[1] <= 64 && dims[0] == dims[1]);
        std::vector<uint64_t> vec;
        for (int i = 0; i < dims[0]; i++) {
           uint64_t val = 0;
           for (int j = 0; j < dims[1]; j++) {
             if (data[i*dims[1]+j]) val |= (1 << j);
           }
           vec.push_back(val);
        }
        return vec;
    }
    else if (dim_num == 1) {
        if (dims[0] == 0) return std::vector<uint64_t>();
        assert(dims[0] <= 64 && dims[0] == 1);
        uint64_t val = 0;
        for (int j = 0; j < dims[0]; j++) {
          if (data[j]) val |= (1 << j);
        }
        return std::vector<uint64_t> {val};
    }
    else {
        std::cerr << "numpy2matrix: Wrong matrix dimension was given" << std::endl;
        exit(EXIT_FAILURE);
    }



}


/**
@brief Wrapper function to call the calculate method of C++ class CZOnePermanentCalculator
@param self A pointer pointing to an instance of the class ZOnePermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
ZOnePermanentCalculator_Wrapper_calculate(ZOnePermanentCalculator_wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", (char*)"gray", (char*)"rows", NULL};


    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;
    int isGray = 0, isRows = 0;
    
    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Opp", kwlist,
                                     &matrix_arg, &isGray, &isRows))
        return Py_BuildValue("");

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return Py_BuildValue("");

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        Py_INCREF(matrix_arg);
    }
    else {
        matrix_arg = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    // create PIC version of the input matrices
    std::vector<uint64_t> matrix_mtx = numpybits2matrix(matrix_arg);

    std::vector<uint64_t> perm = pic::ZOnePermanentCalculator().calculate(matrix_mtx, isGray, isRows);

    // release numpy arrays
    Py_DECREF(matrix_arg);

    //return Py_BuildValue("D", &ret);
    return _PyLong_FromByteArray((const unsigned char*)perm.data(), perm.size() * 8, 1, 0);
}





/**
@brief Wrapper function to call the calculate the Permanent on a DFE
@param self A pointer pointing to an instance of the class ZOnePermanentCalculator_Wrapper.
@param args A tuple of the input arguments: ??????????????
@param kwds A tuple of keywords
*/
static PyObject *
ZOnePermanentCalculator_Wrapper_calculateDFE(ZOnePermanentCalculator_wrapper *self, PyObject *args, PyObject *kwds)
{

    // The tuple of expected keywords
    static char *kwlist[] = {(char*)"matrix", (char*)"sim", (char*)"gray", (char*)"rows", NULL};


    // initiate variables for input arguments
    PyObject *matrix_arg = NULL;
    int isSim = 0, isGray = 0, isRows = 0;
    // parsing input arguments
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Oppp", kwlist,
                                     &matrix_arg, &isSim, &isGray, &isRows))
        return Py_BuildValue("");

    // convert python object array to numpy C API array
    if ( matrix_arg == NULL ) return Py_BuildValue("");

    // establish memory contiguous arrays for C calculations
    if ( PyArray_IS_C_CONTIGUOUS(matrix_arg) ) {
        Py_INCREF(matrix_arg);
    }
    else {
        matrix_arg = PyArray_FROM_OTF(matrix_arg, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
    }

    // create PIC version of the input matrices
    std::vector<uint64_t> matrix_mtx = numpybits2matrix(matrix_arg);

    // initialize DFE array
    initialize_ZOne_DFE();

    std::vector<uint64_t> perm(512);
    pic::ZOnePermanentCalculatorDFE(matrix_mtx, perm, isSim, isGray, isRows);


    // unload DFE
    releive_ZOne_DFE();

    // release numpy arrays
    Py_DECREF(matrix_arg);

    //return Py_BuildValue("D", &perm);
    return _PyLong_FromByteArray((const unsigned char*)perm.data(), perm.size() * 8, 1, 0);
}



static PyGetSetDef ZOnePermanentCalculator_wrapper_getsetters[] = {
    {NULL}  /* Sentinel */
};

/**
@brief Structure containing metadata about the members of class ZOnePermanentCalculator_wrapper.
*/
static PyMemberDef ZOnePermanentCalculator_wrapper_Members[] = {
    {NULL}  /* Sentinel */
};


static PyMethodDef ZOnePermanentCalculator_wrapper_Methods[] = {
    {"calculate", (PyCFunction) ZOnePermanentCalculator_Wrapper_calculate, METH_VARARGS | METH_KEYWORDS,
     "Method to calculate the permanent."
    },
    {"calculateDFE", (PyCFunction) ZOnePermanentCalculator_Wrapper_calculateDFE, METH_VARARGS | METH_KEYWORDS,
     "Method to calculate the permanent on DFE."
    },
    {NULL}  /* Sentinel */
};


/**
@brief A structure describing the type of the class ZOnePermanentCalculator_wrapper.
*/
static PyTypeObject ZOnePermanentCalculator_wrapper_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "ZOnePermanentCalculator_wrapper.ZOnePermanentCalculator_wrapper", /*tp_name*/
  sizeof(ZOnePermanentCalculator_wrapper), /*tp_basicsize*/
  0, /*tp_itemsize*/
  (destructor) ZOnePermanentCalculator_wrapper_dealloc, /*tp_dealloc*/
  #if PY_VERSION_HEX < 0x030800b4
  0, /*tp_print*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b4
  0, /*tp_vectorcall_offset*/
  #endif
  0, /*tp_getattr*/
  0, /*tp_setattr*/
  #if PY_MAJOR_VERSION < 3
  0, /*tp_compare*/
  #endif
  #if PY_MAJOR_VERSION >= 3
  0, /*tp_as_async*/
  #endif
  0, /*tp_repr*/
  0, /*tp_as_number*/
  0, /*tp_as_sequence*/
  0, /*tp_as_mapping*/
  0, /*tp_hash*/
  0, /*tp_call*/
  0, /*tp_str*/
  0, /*tp_getattro*/
  0, /*tp_setattro*/
  0, /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  "Object to represent a Operation_block class of the QGD package.", /*tp_doc*/
  0, /*tp_traverse*/
  0, /*tp_clear*/
  0, /*tp_richcompare*/
  0, /*tp_weaklistoffset*/
  0, /*tp_iter*/
  0, /*tp_iternext*/
  ZOnePermanentCalculator_wrapper_Methods, /*tp_methods*/
  ZOnePermanentCalculator_wrapper_Members, /*tp_members*/
  ZOnePermanentCalculator_wrapper_getsetters, /*tp_getset*/
  0, /*tp_base*/
  0, /*tp_dict*/
  0, /*tp_descr_get*/
  0, /*tp_descr_set*/
  0, /*tp_dictoffset*/
  (initproc) ZOnePermanentCalculator_wrapper_init, /*tp_init*/
  0, /*tp_alloc*/
  ZOnePermanentCalculator_wrapper_new, /*tp_new*/
  0, /*tp_free*/
  0, /*tp_is_gc*/
  0, /*tp_bases*/
  0, /*tp_mro*/
  0, /*tp_cache*/
  0, /*tp_subclasses*/
  0, /*tp_weaklist*/
  0, /*tp_del*/
  0, /*tp_version_tag*/
  #if PY_VERSION_HEX >= 0x030400a1
  0, /*tp_finalize*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b1
  0, /*tp_vectorcall*/
  #endif
  #if PY_VERSION_HEX >= 0x030800b4 && PY_VERSION_HEX < 0x03090000
  0, /*tp_print*/
  #endif
};




} // extern C



#endif //ZOnePermanentCalculator_wrapper
