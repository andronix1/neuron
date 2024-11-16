#include "neuron.h"

void neuron_init(neuron_t *neuron, const size_t size, const func_t activate, func_t activate_derivative) {
	neuron->weights = malloc(sizeof(float) * size);
	neuron->activate = activate;
	neuron->activate_derivative = activate_derivative;
	neuron->size = size;
}

void neuron_random(neuron_t *neuron) {
	neuron->bias = 0.0;
	for (size_t i = 0; i < neuron->size; i++) {
		neuron->weights[i] = randf();
	}
}

neuron_t neuron_new_random(const size_t size, const func_t activate, func_t activate_derivative) {
	neuron_t result;
	neuron_init(&result, size, activate, activate_derivative);
	neuron_random(&result);
	return result;
}

void neuron_free(const neuron_t *neuron) {
	free(neuron->weights);
}

float neuron_predict_non_acivated(const neuron_t *neuron, const float *input) {
	float result = neuron->bias;
	for (size_t i = 0; i < neuron->size; i++) {
		result += neuron->weights[i] * input[i];
	}
	return result;
}

float neuron_predict(const neuron_t *neuron, const float *input) {
	return neuron->activate(neuron_predict_non_acivated(neuron, input));
}

float neuron_predict_derivative(const neuron_t *neuron, const float *input, size_t wid) {
	return neuron->activate_derivative(neuron_predict_non_acivated(neuron, input)) * 
		(wid == neuron->size ? 1 : input[wid]);
}

float neuron_cost(neuron_t *neuron, const train_sample_t *sample) {
	float result = 0.0;
	for (size_t i = 0; i < sample->len; i++) {
		expected_result_t *res = &sample->data[i];
		float prediction = neuron_predict(neuron, res->inputs);
		float delta = prediction - res->outputs[0];
		result += delta * delta / sample->len;
	}
	return result;
}

/*
 * simple_pred(inputs) = SUM(wid) { ws[wid] * inputs[wid] } + bias
 * pred(inputs) = activate(simple_pred(inputs))
 * COST = SUM(pred_id) { (pred(inputs[pred_id]) - res[pred_id]) ** 2 }
 *
 * simple_pred'(inputs) d wid = ws[wid]
 * pred'(inputs) d wid = activate'(simple_pred(inputs)) * simple_pred'(inputs) d wid
 * COST' = SUM(pred_id) {
 *	2 * |pred(inputs[pred_id]) - res[pred_id]| * pred'(inputs[pred_id]) d wid
 * }
*/

float neuron_cost_derivative(neuron_t *neuron, const train_sample_t *sample, size_t wid) {
	float result = 0.0;
	for (size_t i = 0; i < sample->len; i++) {
		expected_result_t *res = &sample->data[i];
		float prediction = neuron_predict(neuron, res->inputs);
		float delta = prediction - res->outputs[0];
		result += delta * neuron_predict_derivative(neuron, res->inputs, wid);
	}
	return result;
}

neuron_cost_derivatives_t neuron_cost_derivatives(neuron_t *neuron, const train_sample_t *sample) {
	float bias_d = neuron_cost_derivative(neuron, sample, neuron->size);
	float *weights_d = malloc(sizeof(float) * neuron->size); // TODO: cache allocated memory or set by ptr	
	for (int i = 0; i < neuron->size; i++) {
		weights_d[i] = neuron_cost_derivative(neuron, sample, i);
	}
	neuron_cost_derivatives_t result = {
		.bias = bias_d,
		.weights = weights_d
	};
	return result;
}

void neuron_cost_derivatives_free(const neuron_cost_derivatives_t *derivatives) {
	free(derivatives->weights);
}

void neuron_train_epoch(neuron_t *neuron, const train_sample_t *sample, const float rate) {
	neuron_cost_derivatives_t derivatives = neuron_cost_derivatives(neuron, sample);
	for (size_t i = 0; i < neuron->size; i++) {
		neuron->weights[i] -= derivatives.weights[i] * rate;
	}
	neuron->bias -= derivatives.bias * rate;
	neuron_cost_derivatives_free(&derivatives);
}

void neuron_train_epochs(neuron_t *neuron, const train_sample_t *sample, const float rate, const size_t epochs) {
	for (size_t i = 0; i < epochs; i++) {
		neuron_train_epoch(neuron, sample, rate);
	}
}

void neuron_print_predictions(const neuron_t *neuron, const train_sample_t *sample) {
	for (size_t i = 0; i < sample->len; i++) {
		expected_result_t *res = &sample->data[i];
		float result = neuron_predict(neuron, res->inputs);
		printf("model(");
		for (size_t j = 0; j < neuron->size; j++) {
			if (j != 0) {
				printf(", ");
			}
			printf("%f", res->inputs[j]);
		}
		printf(") = %f -> %f\n", result, res->outputs[0]);
	}
}
