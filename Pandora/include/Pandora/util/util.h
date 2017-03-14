#pragma once

#include "Pandora/util/random.h"
#include "Pandora/util/type.h"
#include "Pandora/config.h"

namespace Pandora
{

inline serial_size_t conv_out_length(serial_size_t in_length,
                                     serial_size_t window_size,
                                     serial_size_t stride,
                                     padding pad_type)
{
    serial_size_t output_length;

    if (pad_type == padding::same) {
        output_length = in_length;
    } else if (pad_type == padding::valid) {
        output_length = in_length - window_size + 1;
    } else {
        throw error("Not recognized pad_type.");
    }
    return (output_length + stride - 1) / stride;
}

// do nothing
inline void nop()
{
}


}

