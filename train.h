#pragma once

typedef struct {
	float *inputs;
	float output;
} expected_result_t;

typedef struct {
	expected_result_t *results;
	size_t length;	
} train_sample_t;
