#pragma once

#include <algorithm>
#include <numeric>

#include "Pandora/util/util.h"
#include "Pandora/util/type.h"

namespace Pandora {

//inline void vector_div(vec_t &x, float_t denom) {
//#pragma ivdep
//    for(auto iter = x.begin(); iter != x.end() ; ++iter) {
//        *iter /= denom;
//    }
//}

//namespace detail {

//inline void moments_impl_calc_mean(size_t num_examples,
//                                   size_t channels,
//                                   size_t spatial_dim,
//                                   const tensor_t &in,
//                                   vec_t &mean) {
//    for (size_t i = 0; i < num_examples; ++i) {
//        for (size_t j = 0; j < channels; j++) {
//            float_t &rmean = mean[j];
//            const auto it  = in[i].begin() + (j * spatial_dim);
//            rmean          = std::accumulate(it, it + spatial_dim, rmean);
//        }
//    }
//}

//inline void moments_impl_calc_variance(size_t num_examples,
//                                       size_t channels,
//                                       size_t spatial_dim,
//                                       const tensor_t &in,
//                                       const vec_t &mean,
//                                       vec_t &variance) {
//    assert(mean.size() >= channels);
//    for (size_t i = 0; i < num_examples; ++i) {
//        for (size_t j = 0; j < channels; j++) {
//            float_t &rvar    = variance[j];
//            const auto it    = in[i].begin() + (j * spatial_dim);
//            const float_t ex = mean[j];
//            rvar             = std::accumulate(it, it + spatial_dim, rvar,
//                                               [ex](float_t current, float_t x) {
//                    return current + pow(x - ex, float_t{2.0});
//        });
//        }
//    }
//    vector_div(
//                variance,
//                std::max(float_t{1.0f},
//                         static_cast<float_t>(num_examples * spatial_dim) - float_t{1.0f}));
//}

//}  // namespace detail

///**
// * calculate mean/variance across channels
// */
//inline void moments(const tensor_t &in,
//                    size_t spatial_dim,
//                    size_t channels,
//                    vec_t &mean) {
//    const size_t num_examples = static_cast<serial_size_t>(in.size());
//    assert(in[0].size() == spatial_dim * channels);

//    mean.resize(channels);
//    std::fill(mean.begin(), mean.end(), float_t{0.0});
//    detail::moments_impl_calc_mean(num_examples, channels, spatial_dim, in, mean);
//    vector_div(mean, (float_t)num_examples * spatial_dim);
//}

//inline void moments(const tensor_t &in,
//                    size_t spatial_dim,
//                    size_t channels,
//                    vec_t &mean,
//                    vec_t &variance) {
//    const size_t num_examples = static_cast<serial_size_t>(in.size());
//    assert(in[0].size() == spatial_dim * channels);

//    // calc mean
//    moments(in, spatial_dim, channels, mean);

//    variance.resize(channels);
//    std::fill(variance.begin(), variance.end(), float_t{0.0});
//    detail::moments_impl_calc_variance(num_examples, channels, spatial_dim, in,
//                                       mean, variance);
//}




}
