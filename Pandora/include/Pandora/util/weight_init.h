#pragma once

#include "Pandora/util/util.h"
#include "Pandora/util/random.h"

namespace Pandora
{
namespace weight_init
{

class function {
public:
    virtual void fill(vec_t *weight,
                    serial_size_t fan_in,
                    serial_size_t fan_out) = 0;
};

class scalable : public function {
public:
    explicit scalable(float_t value) : m_scale(value) {}

    void scale(float_t value) { m_scale = value; }

protected:
    float_t m_scale;
};

class xavier : public scalable {
public:
    xavier() : scalable(float_t(6)) {}
    explicit xavier(float_t value) : scalable(value) {}

    void fill(vec_t *weight,
            serial_size_t fan_in,
            serial_size_t fan_out) override
    {
        const float_t weight_base = std::sqrt(m_scale / (fan_in + fan_out));

        uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
    }
};

class lecun : public scalable {
public:
    lecun() : scalable(float_t(1)) {}
    explicit lecun(float_t value) : scalable(value) {}

    void fill(vec_t *weight,
            serial_size_t fan_in,
            serial_size_t) override {

        const float_t weight_base = m_scale / std::sqrt(float_t(fan_in));

        uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
    }
};

class gaussian : public scalable {
public:
    gaussian() : scalable(float_t(1)) {}
    explicit gaussian(float_t sigma) : scalable(sigma) {}

    void fill(vec_t *weight,
            serial_size_t,
            serial_size_t) override {

        gaussian_rand(weight->begin(), weight->end(), float_t{0}, m_scale);
    }
};

class constant : public scalable {
public:
    constant() : scalable(float_t(0)) {}
    explicit constant(float_t value) : scalable(value) {}

    void fill(vec_t *weight,
            serial_size_t,
            serial_size_t) override {

        std::fill(weight->begin(), weight->end(), m_scale);
    }
};

class he : public scalable {
public:
    he() : scalable(float_t(2)) {}
    explicit he(float_t value) : scalable(value) {}

    void fill(vec_t *weight,
            serial_size_t fan_in,
            serial_size_t) override {

        const float_t sigma = std::sqrt(m_scale / fan_in);

        gaussian_rand(weight->begin(), weight->end(), float_t{0}, sigma);
    }
};


}}

