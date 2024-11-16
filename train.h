#pragma once

#include "array.h"

typedef struct {
	float *inputs;
	float *outputs;
} expected_result_t;

inline expected_result_t expected_result(float *inputs, float *outputs) {
	expected_result_t result = { .inputs = inputs, .outputs = outputs };
	return result;
}

define_array_type(expected_result_t, train_sample);
