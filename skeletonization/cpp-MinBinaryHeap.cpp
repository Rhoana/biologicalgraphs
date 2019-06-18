#ifndef __CPP_MIN_BINARY_HEAP_CPP__
#define __CPP_MIN_BINARY_HEAP_CPP__

#include "cpp-MinBinaryHeap.h"
#include <stdlib.h>




////////////////////////////////////////////////////////////////////////
// Constructor/destructor functions
////////////////////////////////////////////////////////////////////////

template <class PtrType>
MinBinaryHeap<PtrType>::
MinBinaryHeap(PtrType base, double *value_ptr, int nentries) :
NMAX(nentries),
N(0)
{
   // compute offsets to data entries in struct referenced by PtrType
   if (value_ptr) value_offset = (unsigned char *)value_ptr - (unsigned char *)base;

   // the keys in the heap
   keys = new PtrType[NMAX];

   // ordered indices indicate location in key
   pq = new int[NMAX + 1];

   // is index i in the priority queue
   qp = new int[NMAX];
   for (int i = 0; i < NMAX; ++i)
      qp[i] = -1;
}



template <class PtrType>
MinBinaryHeap<PtrType>::
~MinBinaryHeap(void)
{
   // delete all the created arrays
   delete[] keys;
   delete[] pq;
   delete[] qp;
}



////////////////////////////////////////////////////////////////////////
// Attribute functions
////////////////////////////////////////////////////////////////////////

template <class PtrType>
int MinBinaryHeap<PtrType>::
Size(void) const
{
   // return the number of keys in the priority queue
   return N;
}



template <class PtrType>
PtrType MinBinaryHeap<PtrType>::
MinKey(void) const
{
   // return the minimum key
   return keys[pq[1]];
}



template <class PtrType>
PtrType MinBinaryHeap<PtrType>::
KeyOf(int i) const
{
   // return the key associated with index i
   return keys[i];
}



////////////////////////////////////////////////////////////////////////
// Property functions
////////////////////////////////////////////////////////////////////////

template <class PtrType>
int MinBinaryHeap<PtrType>::
IsEmpty(void) const
{
   // is the binary heap empty
   if (N == 0) return 1;
   else return 0;
}



template <class PtrType>
int MinBinaryHeap<PtrType>::
Contains(int i) const
{
   // is this index in the priority queue
   if (qp[i] == -1) return 0;
   else return 1;
}



////////////////////////////////////////////////////////////////////////
// Manipulation functions
////////////////////////////////////////////////////////////////////////

template <class PtrType>
void MinBinaryHeap<PtrType>::
Insert(int i, PtrType key)
{
   // insert the key with index i
   N++;
   qp[i] = N;
   pq[N] = i;
   keys[i] = key;
   Swim(N);
}



template <class PtrType>
int MinBinaryHeap<PtrType>::
MinIndex(void) const
{
   // return the index associated with the minimum key
   return pq[1];
}



template <class PtrType>
PtrType MinBinaryHeap<PtrType>::
DeleteMin(void)
{
   // remove a minimum key and return its associated index
   int min = pq[1];
   Exch(1, N--);
   Sink(1);
   qp[min] = -1;
   PtrType ptr = keys[min];
   keys[min] = NULL;
   return ptr;
}



template <class PtrType>
void MinBinaryHeap<PtrType>::
Delete(int  i)
{
   // remove the key associated with index i
   int index = qp[i];
   Exch(index, N--);
   Swim(index);
   Sink(index);
   keys[i] = NULL;
   qp[i] = -1;
}



template <class PtrType>
void MinBinaryHeap<PtrType>::
ChangeKey(int i, PtrType key)
{
   // change the key associated with index i to the specified value
   keys[i] = key;
   Swim(qp[i]);
   Sink(qp[i]);
}



template <class PtrType>
void MinBinaryHeap<PtrType>::
DecreaseKey(int i, PtrType key)
{
   // decrease the key associated with index i to the specified value
   keys[i] = key;
   Swim(qp[i]);
}



template <class PtrType>
void MinBinaryHeap<PtrType>::
IncreaseKey(int i, PtrType key)
{
   // increase the key associated with index i to the specified value
   keys[i] = key;
   Sink(qp[i]);
}



////////////////////////////////////////////////////////////////////////
// Internal functions
////////////////////////////////////////////////////////////////////////

template <class PtrType>
int MinBinaryHeap<PtrType>::
Greater(int i, int j) const
{
   if (Compare(i, j) > 0) return 1;
   else return 0;
}



template <class PtrType>
void MinBinaryHeap<PtrType>::
Exch(int i, int j)
{
   int swap = pq[i]; pq[i] = pq[j]; pq[j] = swap;
   qp[pq[i]] = i; qp[pq[j]] = j;
}



template <class PtrType>
double MinBinaryHeap<PtrType>::
Value(int i) const
{
   return *((double *)((unsigned char *)keys[i] + value_offset));
}



template <class PtrType>
int MinBinaryHeap<PtrType>::
Compare(int i, int j) const
{
   // get values
   double value1 = Value(pq[i]);
   double value2 = Value(pq[j]);

   // compare values
   if (value1 < value2) return -1;
   else if (value1 > value2) return 1;
   else return 0;
}



template <class PtrType>
void MinBinaryHeap<PtrType>::
Swim(int k)
{
   // swim this index to its new position
   while (k > 1 && Greater(k / 2, k)) {
      Exch(k, k / 2);
      k = k / 2;
   }
}



template <class PtrType>
void MinBinaryHeap<PtrType>::
Sink(int k)
{
   // sink this index  to its new position
   while (2 * k <= N) {
      int j = 2 * k;
      if (j < N && Greater(j, j + 1)) j++;
      if (!Greater(k, j)) break;
      Exch(k, j);
      k = j;
   }
}

#endif