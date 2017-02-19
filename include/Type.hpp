#ifndef TYPE_H
#define TYPE_H

#include "vector"
#include <iostream>
#include "math.h"
#include "time.h"
#include "omp.h"
#include "cilk/reducer.h"
#include "cilk/cilk.h"
#include "cilk/reducer_opadd.h"

namespace DeepLearning
{
    // columnwise template matrix class
    template <typename type>
    class mat
    {
    public:
        typedef std::vector<type> typeVec;
    private:
        typeVec m_data;
        int     m_nrow, m_ncol;

    public:
        mat() : m_data(), m_ncol(0), m_nrow(0) {}
        mat(int nrow, int ncol) : m_data(nrow * ncol), m_ncol(ncol), m_nrow(nrow) {}

        type*       operator[](int row) { return m_data.data() + row * m_ncol; }
        typeVec&    data(void) { return m_data; }
        int         nrow(void) { return m_nrow; }
        int         ncol(void) { return m_ncol; }
        void        resize(int nrow, int ncol) {
            m_data.resize(nrow* ncol);
            m_ncol = ncol;
            m_nrow = nrow;
        }
    };

    class randomGenerator
    {
    private:
        unsigned long    m_seed;

    public:
        randomGenerator() { m_seed = 850702123123 + omp_get_wtime(); }

        unsigned long getSeed() { return m_seed; }
        unsigned long getRand() { return ((m_seed = 214013 * m_seed + 2531011) & 4294967295); }
        double getUniform() {
            return getRand() / 4294967296.0;
        }
        double boxMuller()  {
            return sqrt(-2.0 * log(getUniform())) * cos(2.0 * 3.141592 * getUniform());
        }
    };

    typedef std::vector<double> rVec;
    typedef mat<double>         rMat;
    typedef std::vector<int>    iVec;
    typedef mat<int>            iMat;
}

#endif // TYPE_H
