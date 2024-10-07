#pragma once

#include "array.h"

typedef struct {
	float *inputs;
	float output;
} expected_result_t;

inline expected_result_t expected_result(float *inputs, float output) {      
	expected_result_t result = { .inputs = inputs, .output = output };
	return result;
}

define_array_type(expected_result_t, train_sample);
