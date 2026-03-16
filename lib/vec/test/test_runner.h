#pragma once
#include "vec.cuh"
#include <random>

inline std::mt19937 rng(std::random_device{}());
inline std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
constexpr float epsilon = 2e-6;

template<size_t dim>
vec<dim> init_vec() {
    vec<dim> v;
    for (size_t i = 0; i < dim; ++i)
        v.data[i] = dist(rng);
    return v;
}

template<template<size_t> class Test, size_t start, size_t end>
void run_tests() {
    if constexpr (start <= end) {
        Test<start>{}();
        run_tests<Test, start + 1, end>();
    }
}
