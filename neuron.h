#pragma once

#include <stdint.h>
#include <malloc.h>
#include "train.h"
#include "mathf.h"

typedef struct {
	float *weights;
	float bias;
	size_t size;
	func_t activate;
	func_t activate_derivative;
} neuron_t;

neuron_t neuron_new_random(const size_t size, const func_t activate, func_t activate_derivative);
void neuron_init(neuron_t *neuron, const size_t size, const func_t activate, func_t activate_derivative);
void neuron_random(neuron_t *neuron);
void neuron_free(const neuron_t *neuron);
float neuron_predict(const neuron_t *neuron, const float *input);
float neuron_cost(neuron_t *neuron, const train_sample_t *sample);

typedef struct {
	float *weights;
	float bias;
} neuron_cost_derivatives_t;

neuron_cost_derivatives_t neuron_cost_derivatives(neuron_t *neuron, const train_sample_t *sample);
void neuron_cost_derivatives_free(const neuron_cost_derivatives_t *derivatives);
void neuron_train_epoch(neuron_t *neuron, const train_sample_t *sample, const float rate);
void neuron_train_epochs(neuron_t *neuron, const train_sample_t *sample, const float rate, const size_t epochs);
void neuron_print_predictions(const neuron_t *neuron, const train_sample_t *sample);
