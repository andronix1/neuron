#include "layer.h"

impl_array_type(size_t, layer_ids)
impl_array_simple_free(layer_ids)

void layer_neuron_free(layer_neuron_t *neuron) {
	layer_ids_free(&neuron->input_ids);
}

impl_array_type(layer_neuron_t, layer)
impl_array_nested_free(layer, layer_neuron_free)

impl_array_type(layer_t, model_layers)
impl_array_nested_free(model_layers, layer_free)

float *multilayer_model_predict(const multilayer_model_t *model, const float *input) {
	size_t max_layer_size = model->outputs.len;
	size_t max_inputs_size = 0;
	for (size_t i = 0; i < model->layers.len; i++) {
		size_t layer_size = model->layers.data[i].len;
		if (layer_size > max_layer_size) {
			max_layer_size = layer_size;
		}
		const layer_t *layer = &model->layers.data[i];
		for (size_t j = 0; j < layer->len; j++) {
			size_t inputs_size = layer->data[j].neuron.size;
			if (inputs_size > max_inputs_size) {
				max_inputs_size = inputs_size;
			}
		}
	}

	float *prev_values = alloca(sizeof(float) * max_layer_size),
	      *cur_values = alloca(sizeof(float) * max_layer_size),
	      *inputs = alloca(sizeof(float) * max_inputs_size);

	memcpy(prev_values, input, sizeof(float) * model->layers.data[0].len);

	for (size_t lid = 0; lid < model->layers.len; lid++) {
		if (lid != 0) {
			memcpy(prev_values, cur_values, sizeof(float) * max_layer_size);
		}
		const layer_t *layer = &model->layers.data[lid];
		for (size_t i = 0; i < layer->len; i++) {	
			const layer_neuron_t *layer_neuron = &layer->data[i];
			for (int j = 0; j < layer_neuron->input_ids.len; j++) {
				inputs[j] = prev_values[layer_neuron->input_ids.data[j]];
			}
			cur_values[i] = neuron_predict(&layer_neuron->neuron, inputs);
		}	
	}
	
	float *result = malloc(sizeof(float) * model->outputs.len);
	for (size_t i = 0; i < model->outputs.len; i++) {
		result[i] = cur_values[model->outputs.data[i]];
	}

	return result;
}

void multilayer_model_free(const multilayer_model_t *model) {	
	model_layers_free(&model->layers);
	layer_ids_free(&model->outputs);
}
