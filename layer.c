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

float multilayer_model_cost(multilayer_model_t *model, const train_sample_t *sample) {
	float result = 0.0;
	for (size_t i = 0; i < sample->len; i++) {
		expected_result_t *res = &sample->data[i];
		float *prediction = multilayer_model_predict(model, res->inputs);
		for (size_t j = 0; j < model->outputs.len; j++) {
			float delta = prediction[j] - res->outputs[j];
			result += delta * delta / sample->len;
		}
		free(prediction);
	}
	return result;
}

void multilayer_model_free(const multilayer_model_t *model) {	
	model_layers_free(&model->layers);
	layer_ids_free(&model->outputs);
}

#define mm_eps_der 1e-3

// TODO: better derivative
float multilayer_model_derivative(multilayer_model_t *model, const train_sample_t *sample, size_t layer_id, size_t neuron_id, size_t wid) {
	float a = multilayer_model_cost(model, sample);
	float *w = &model->layers.data[layer_id].data[neuron_id].neuron.weights[wid];
	*w += mm_eps_der;
	float b = multilayer_model_cost(model, sample);
	*w -= mm_eps_der;
	return (a - b) / mm_eps_der;
}

void multilayer_model_train_epoch(multilayer_model_t *model, const train_sample_t *sample, const float rate) {
	for (size_t lid = 0; lid < model->layers.len; lid++) {
		layer_t *layer = &model->layers.data[lid];
		for (size_t nid = 0; nid < layer->len; nid++) {
			layer_neuron_t *neuron = &layer->data[nid];
			for (size_t wid = 0; wid < neuron->neuron.size; wid++) {
				neuron->neuron.weights[wid] += multilayer_model_derivative(model, sample, lid, nid, wid) * rate;
			}
		}
	}

}

void multilayer_model_train_epochs(multilayer_model_t *model, const train_sample_t *sample, const float rate, size_t epochs) {
	for (size_t i = 0; i < epochs; i++) {
		multilayer_model_train_epoch(model, sample, rate);
	}
}
