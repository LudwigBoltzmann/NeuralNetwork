#pragma once

#include <vector>
#include <limits>
#include "Pandora/config.h"
#include "Pandora/util/error.h"

namespace Pandora
{

template <typename T>
class index3d {
public:
    index3d(T width, T height, T channel)
    {
        reshape(width, height, channel);
    }

    index3d(T width, T height, T depth, T channel)
    {
        reshape(width, height, depth, channel);
    }

    index3d()
        : numWidth(0), numHeight(0), numDepth(0), numChannel(0)
    { }

    void reshape(T width, T height, T channel)
    {
        numWidth  = width;
        numHeight = height;
        numDepth = 1;
        numChannel  = channel;

        if ((int64_t)width * height * channel > std::numeric_limits<T>::max()) {
            throw error(format_str(
            "error while constructing layer: layer size too large for "
            "Gaia\nWidthxHeightxChannels=%dx%dx%d >= max size of "
            "[%s](=%d)",
            width, height, channel, typeid(T).name(), std::numeric_limits<T>::max()));
        }
    }

    void reshape(T width, T height, T depth, T channel)
    {
        numWidth  = width;
        numHeight = height;
        numDepth = depth;
        numChannel  = channel;

        if ((int64_t)width * height * depth * channel > std::numeric_limits<T>::max()) {
            throw error(format_str(
            "error while constructing layer: layer size too large for "
            "Gaia\nWidthxHeightxDepthxChannels=%dx%dx%dx%d >= max size of "
            "[%s](=%d)",
            width, height, depth, channel, typeid(T).name(), std::numeric_limits<T>::max()));
        }
    }

    T get_index(T x, T y, T channel) const
    {
        assert(x >= 0 && x < numWidth);
        assert(y >= 0 && y < numHeight);
        assert(channel >= 0 && channel < numChannel);
        return (numHeight * channel + y) * numWidth + x;
    }

    T get_index(T x, T y, T z, T channel) const
    {
        assert(x >= 0 && x < numWidth);
        assert(y >= 0 && y < numHeight);
        assert(z >= 0 && z < numDepth);
        assert(channel >= 0 && channel < numChannel);
//        return channel * numWidth * numHeight * numDepth +
//                z * numWidth * numHeight +
//                y * numWidth +
//                x;
        return ((numDepth * channel + z) * numHeight + y) * numWidth + x;
    }


    T area() const
    {
        return numWidth * numHeight;
    }

    T volume() const
    {
        return numWidth * numHeight * numDepth;
    }

    T size() const
    {
        return numWidth * numHeight * numDepth * numChannel;
    }

    T numWidth;
    T numHeight;
    T numDepth;
    T numChannel;
};

typedef index3d<serial_size_t> shape3d;

template <typename T>
bool operator==(const index3d<T> &lhs, const index3d<T> &rhs)
{
    return  (lhs.numWidth   ==   rhs.numWidth)  &&
            (lhs.numHeight  ==   rhs.numHeight) &&
            (lhs.numDepth   ==   rhs.numDepth)  &&
            (lhs.numChannel ==   rhs.numChannel);
}

template <typename T>
bool operator!=(const index3d<T> &lhs, const index3d<T> &rhs)
{
    return !(lhs == rhs);
}

template <typename stream, typename T>
stream &operator<<(stream &s, const index3d<T> &d) {
  s << d.numWidth << "x" << d.numHeight << "x" << d.numDepth<< "x" << d.numChannel;
  return s;
}

template <typename stream, typename T>
stream &operator<<(stream &s, const std::vector<index3d<T>> &d) {
    s << "[";
    for (serial_size_t i = 0; i < d.size(); ++i) {
        if (i) s << ",";
        s << "[" << d[i] << "]";
    }
    s << "]";
    return s;
}
}
