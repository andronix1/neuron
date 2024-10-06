#pragma once

#include <stdint.h>
#include <malloc.h>
#include "train.h"
#include "func.h"
#include "rand.h"

typedef struct {
	float *weights;
	float bias;
	size_t size;
	func_t activate;
} neuron_t;

void neuron_init(neuron_t *neuron, const size_t size, func_t activate);
void neuron_random(neuron_t *neuron);
void neuron_free(neuron_t *neuron);

float neuron_predict(neuron_t *neuron, const float *input);
float neuron_cost(neuron_t *neuron, train_sample_t *sample);

typedef struct {
	float *weights;
	float bias;
} neuron_cost_derivatives_t;

void neuron_train_epoch(neuron_t *neuron, train_sample_t *sample, float rate);
void neuron_print_predictions(neuron_t *neuron, train_sample_t *sample);
