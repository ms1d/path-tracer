#pragma once
#include "mat.cuh"
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

template<size_t r, size_t c>
mat<r, c> init_mat() {
    mat<r, c> m;
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            m.data[i][j] = dist(rng);
    return m;
}

template<template<size_t, size_t, size_t, size_t> class Test, size_t r1_start, size_t r1_end, size_t c1_start, size_t c1_end, size_t r2_start, size_t r2_end, size_t c2_start, size_t c2_end>
void run_tests() {
	const size_t c1_start_og = c1_start;
	const size_t r2_start_og = r2_start;
    const size_t c2_start_og = c2_start;

	if constexpr (r1_start <= r1_end) {
		if constexpr (c1_start <= c1_end) {
			if constexpr (r2_start <= r2_end) {
				if constexpr (c2_start <= c2_end) {
					Test<r1_start, c1_start, r2_start, c2_start + 1>{}();
				}
				
				Test<r1_start, c1_start, r2_start + 1, c2_start_og>{}();
			}
			
			Test<r1_start, c1_start + 1, r2_start_og, c2_start_og>{}();
		}
		
		Test<r1_start + 1, c1_start_og, r2_start_og, c2_start_og>{}();
	}
}
