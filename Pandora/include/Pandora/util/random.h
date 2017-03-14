#pragma once


#include <limits>
#include <random>
#include <type_traits>

#include "Pandora/util/error.h"
#include "Pandora/config.h"

namespace Pandora
{
class random_generator {
private:
    random_generator() : m_gen(1) {}
    std::mt19937 m_gen;

public:
    static random_generator &get_instance() {
        static random_generator instance;
        return instance;
    }

    std::mt19937 &operator()() { return m_gen; }

    void set_seed(unsigned int seed) { m_gen.seed(seed); }
};

template <typename T>
inline typename std::enable_if<std::is_integral<T>::value, T>::type uniform_rand(T min, T max)
{
    std::uniform_int_distribution<T> dst(min, max);
    return dst(random_generator::get_instance()());
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type uniform_rand(T min, T max)
{
    std::uniform_real_distribution<T> dst(min, max);
    return dst(random_generator::get_instance()());
}

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type gaussian_rand(T mean, T sigma)
{
    std::normal_distribution<T> dst(mean, sigma);
    return dst(random_generator::get_instance()());
}

inline void set_random_seed(unsigned int seed)
{
    random_generator::get_instance().set_seed(seed);
}

template <typename Container>
inline int uniform_idx(const Container &t)
{
    return uniform_rand(0, int(t.size() - 1));
}

inline bool bernoulli(float_t p)
{
    return uniform_rand(float_t{0}, float_t{1}) <= p;
}

template <typename iter, typename T>
inline void uniform_rand(iter begin, iter end, T min, T max)
{
    for(iter it = begin; it != end; ++it)
        *it = uniform_rand(min, max);
}

template <typename iter, typename T>
inline void gaussian_rand(iter begin, iter end, T mean, T sigma)
{
    for(iter it = begin; it != end; ++it)
        *it = gaussian_rand(mean, sigma);
}

}
