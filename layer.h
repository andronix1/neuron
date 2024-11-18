#pragma once

#include <string.h>
#include "neuron.h"
#include "array.h"

define_array_type(size_t, layer_ids)

typedef struct {
	const neuron_t neuron;
	const layer_ids_t input_ids;
} layer_neuron_t;

void layer_neuron_free(layer_neuron_t *neuron);

inline layer_neuron_t layer_neuron(neuron_t neuron, layer_ids_t input_ids) {
	layer_neuron_t result = {
		.neuron = neuron,
		.input_ids = input_ids
	};
	return result;
}

define_array_type(layer_neuron_t, layer)

define_array_type(layer_t, model_layers);

typedef struct {
	const model_layers_t layers;
	const layer_ids_t outputs;
} multilayer_model_t;

#define multilayer_model(_layers, _outputs) { .layers = _layers, .outputs = _outputs }

float *multilayer_model_predict(const multilayer_model_t *model, const float *input);
void multilayer_model_free(const multilayer_model_t *model);
float multilayer_model_cost(multilayer_model_t *model, const train_sample_t *sample);
void multilayer_model_train_epoch(multilayer_model_t *model, const train_sample_t *sample, const float rate);
void multilayer_model_train_epochs(multilayer_model_t *model, const train_sample_t *sample, const float rate, size_t epochs); 
