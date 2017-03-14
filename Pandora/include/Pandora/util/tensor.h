#pragma once

#include <vector>
#include "Pandora/util/aligned_allocator.h"
#include "Pandora/config.h"

namespace Pandora
{

class tensor
{
public:
    typedef std::vector<float_t,aligned_allocator<float_t,64>>    rVec_type;
    typedef std::vector<float_t,aligned_allocator<float_t,64>>    iVec_type;
    typedef rVec_type::iterator                                 iterator;
    typedef rVec_type::const_iterator                           const_iterator;

    /// if dim.size == 4    if dim.size == 3    if dim.size == 2    if dim.size == 1
    /// 0 : width           0 : width           0 : width           0 : width
    /// 1 : height          1 : height          1 : height or channel
    /// 2 : depth           2 : depth or channel
    /// 3 : channel
    iVec_type   dim;

private:
    rVec_type   m_data;

public:
    tensor()
        : m_data(), dim() {}
    tensor(iVec_type dims)
        : m_data(), dim(dims)
    {
        size_t size = 1;
        for(int i = 0; i < dim.size(); ++i) size *= dim[i];
        m_data.resize(size);
    }
    tensor(const tensor& other)
        : m_data(other.m_data), dim(other.dim) {}
    tensor(tensor&& other)
        : m_data(std::move(other.m_data)),
          dim(std::move(other.dim))
    {}

    rVec_type&  getData(void) { return m_data; }
    size_t  offset(int w, int h = 0, int d = 0, int c = 0) {
        switch (dim.size()) {
        case 4:     return ((c * dim[2] + d) * dim[1] + h) * dim[0] + w;
        case 3:     return ((d) * dim[1] + h) * dim[0] + w;
        case 2:     return (h) * dim[0] + w;
        case 1:     return w;
        default:    return 0;
        }
    }

    tensor& operator()(const tensor& other)
    {
        m_data = other.m_data;
        dim = other.dim;
        return *this;
    }

    float_t& operator()(int w, int h = 0, int d = 0, int c = 0) {
        return m_data[offset(w,h,d,c)];
    }

    iterator        begin(void)         { return m_data.begin(); }
    const_iterator  begin(void) const   { return m_data.begin(); }
    iterator        end(void)           { return m_data.end(); }
    const_iterator  end(void) const     { return m_data.end(); }

};

inline void fill_tensor(tensor& data, float_t val)
{
    std::fill(data.begin(),data.end(),val);
}

inline void fill_tensor(tensor& data, float_t val, size_t width, size_t height = 1, size_t depth = 1, size_t channel = 1)
{
    data.getData().resize(width * height * depth * channel, val);
    data.dim.resize(4);
    data.dim[0] = width;
    data.dim[1] = height;
    data.dim[2] = depth;
    data.dim[3] = channel;
}


}
