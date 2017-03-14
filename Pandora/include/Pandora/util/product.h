#pragma once


namespace Pandora {

template <typename T>
struct generic_vec_type {
    typedef T register_type;
    typedef T value_type;
    enum { unroll_size = 1 };
    static register_type set1(const value_type &x) { return x; }
    static register_type zero() { return register_type(0); }
    static register_type mul(const register_type &v1, const register_type &v2) {
        return v1 * v2;
    }
    static register_type add(const register_type &v1, const register_type &v2) {
        return v1 + v2;
    }
    static register_type madd(const register_type &v1,
                              const register_type &v2,
                              const register_type &v3) {
        return v1 * v2 + v3;
    }
    static register_type load(const value_type *px) { return *px; }
    static register_type loadu(const value_type *px) { return *px; }
    static void store(value_type *px, const register_type &v) { *px = v; }
    static void storeu(value_type *px, const register_type &v) { *px = v; }
    static value_type resemble(const register_type &x) { return x; }
};

// generic dot-product
template <typename T>
inline typename T::value_type dot_product_nonaligned(
        const typename T::value_type *f1,
        const typename T::value_type *f2,
        std::size_t size) {
    typename T::register_type result = T::zero();

    for (std::size_t i = 0; i < size / T::unroll_size; ++i)
        result = T::madd(T::loadu(&f1[i * T::unroll_size]),
                T::loadu(&f2[i * T::unroll_size]), result);

    typename T::value_type sum = T::resemble(result);

    for (std::size_t i = (size / T::unroll_size) * T::unroll_size; i < size; ++i)
        sum += f1[i] * f2[i];

    return sum;
}
template <typename T>
inline typename T::value_type dot_product_aligned(
        const typename T::value_type *f1,
        const typename T::value_type *f2,
        std::size_t size) {
    typename T::register_type result = T::zero();

    assert(is_aligned(T(), f1));
    assert(is_aligned(T(), f2));

    for (std::size_t i = 0; i < size / T::unroll_size; ++i)
        result = T::madd(T::load(&f1[i * T::unroll_size]),
                T::load(&f2[i * T::unroll_size]), result);

    typename T::value_type sum = T::resemble(result);

    for (std::size_t i = (size / T::unroll_size) * T::unroll_size; i < size; ++i)
        sum += f1[i] * f2[i];

    return sum;
}


template <typename T>
inline void muladd_nonaligned(const typename T::value_type *src,
                              typename T::value_type c,
                              std::size_t size,
                              typename T::value_type *dst) {
    typename T::register_type factor = T::set1(c);

    for (std::size_t i = 0; i < size / T::unroll_size; ++i) {
        typename T::register_type d = T::loadu(&dst[i * T::unroll_size]);
        typename T::register_type s = T::loadu(&src[i * T::unroll_size]);
        T::storeu(&dst[i * T::unroll_size], T::madd(s, factor, d));
    }

    for (std::size_t i = (size / T::unroll_size) * T::unroll_size; i < size; ++i)
        dst[i] += src[i] * c;
}
template <typename T>
inline void muladd_aligned(const typename T::value_type *src,
                           typename T::value_type c,
                           std::size_t size,
                           typename T::value_type *dst) {
    typename T::register_type factor = T::set1(c);

    for (std::size_t i = 0; i < size / T::unroll_size; ++i) {
        typename T::register_type d = T::load(&dst[i * T::unroll_size]);
        typename T::register_type s = T::load(&src[i * T::unroll_size]);
        T::store(&dst[i * T::unroll_size], T::madd(s, factor, d));
    }

    for (std::size_t i = (size / T::unroll_size) * T::unroll_size; i < size; ++i)
        dst[i] += src[i] * c;
}

template <typename T>
inline void reduce_nonaligned(const typename T::value_type *src,
                              std::size_t size,
                              typename T::value_type *dst) {
    for (std::size_t i = 0; i < size / T::unroll_size; ++i) {
        typename T::register_type d = T::loadu(&dst[i * T::unroll_size]);
        typename T::register_type s = T::loadu(&src[i * T::unroll_size]);
        T::storeu(&dst[i * T::unroll_size], T::add(d, s));
    }

    for (std::size_t i = (size / T::unroll_size) * T::unroll_size; i < size; ++i)
        dst[i] += src[i];
}
template <typename T>
inline void reduce_aligned(const typename T::value_type *src,
                           std::size_t size,
                           typename T::value_type *dst) {
    for (std::size_t i = 0; i < size / T::unroll_size; ++i) {
        typename T::register_type d = T::loadu(&dst[i * T::unroll_size]);
        typename T::register_type s = T::loadu(&src[i * T::unroll_size]);
        T::storeu(&dst[i * T::unroll_size], T::add(d, s));
    }

    for (std::size_t i = (size / T::unroll_size) * T::unroll_size; i < size; ++i)
        dst[i] += src[i];
}


}
