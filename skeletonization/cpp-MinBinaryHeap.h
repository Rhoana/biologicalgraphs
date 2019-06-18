#ifndef __CPP_MIN_BINARY_HEAP__
#define __CPP_MIN_BINARY_HEAP__

/////////////////////////////////////////////////////////////////////
// CLASS DEFINITION
/////////////////////////////////////////////////////////////////////

template <class PtrType>
class MinBinaryHeap {
public:
    //////////////////////////////////
    //// CONSTRUCTORS/DESTRUCTORS ////
    //////////////////////////////////

    MinBinaryHeap(PtrType base, double *value_ptr, int nentries);
    ~MinBinaryHeap();


    /////////////////////////////
    //// ATTRIBUTE FUNCTIONS ////
    /////////////////////////////

    int Size(void) const;
    int MinIndex(void) const;
    PtrType MinKey(void) const;
    PtrType KeyOf(int i) const;
    

    ////////////////////////////
    //// PROPERTY FUNCTIONS ////
    ////////////////////////////

    int IsEmpty(void) const;
    int Contains(int i) const;

    
    ////////////////////////////////
    //// MANIPULATION FUNCTIONS ////
    ////////////////////////////////
    
    // insert functions
    void Insert(int i, PtrType key);

    // key manipulation functions
    void ChangeKey(int i, PtrType key);
    void DecreaseKey(int i, PtrType key);
    void IncreaseKey(int i, PtrType key);
    
    // deletion functions
    PtrType DeleteMin(void);
    void Delete(int i);


    ////////////////////////////////////////////////////////////////////////
    // INTERNAL STUFF BELOW HERE
    ////////////////////////////////////////////////////////////////////////
   
private:
    // value property functions
    double Value(int i) const;
    int Compare(int i, int j) const;
    int Greater(int i, int j) const;

    // heap manipulation helper functions
    void Exch(int i, int j);
    void Swim(int k);
    void Sink(int k);

private:
    // instance variables
    int NMAX;
    int N;
    int *pq;
    int *qp;
    PtrType *keys;
    int value_offset;
};

#include "cpp-MinBinaryHeap.cpp"

#endif