#pragma once

#include <vector>
#include <cassert>
#include <functional>
#include <limits>
#include <random>
#include <cstdarg>
#include <cstdio>
#include <sstream>
#include <string>
#include <type_traits>
#include <memory>
#include <algorithm>

#include "Pandora/util/aligned_allocator.h"
#include "Pandora/util/tensor.h"


namespace Pandora
{

enum class net_phase
{
    train,
    test
};

enum class padding
{
    valid,  ///< use valid pixels of input
    same    ///< add zero-padding around input so as to keep image size
};

typedef std::uint32_t                                       serial_size_t;
typedef serial_size_t                                       label_t;
typedef serial_size_t                                       layer_size_t;
typedef std::vector<float_t,aligned_allocator<float_t,64>>  vec_t;
typedef tensor                                              tensor_t;
}
