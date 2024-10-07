#include "neuron.h"

void neuron_init(neuron_t *neuron, const size_t size, const func_t activate) {
	neuron->weights = malloc(sizeof(float) * size);
	neuron->activate = activate;
	neuron->size = size;
}

void neuron_random(neuron_t *neuron) {
	neuron->bias = 0.0;
	for (size_t i = 0; i < neuron->size; i++) {
		neuron->weights[i] = randf();
	}
}

neuron_t neuron_new_random(const size_t size, const func_t activate) {
	neuron_t result;
	neuron_init(&result, size, activate);
	neuron_random(&result);
	return result;
}

void neuron_free(const neuron_t *neuron) {
	free(neuron->weights);
}

float neuron_predict(const neuron_t *neuron, const float *input) {
	float result = neuron->bias;
	for (size_t i = 0; i < neuron->size; i++) {
		result += neuron->weights[i] * input[i];
	}
	return neuron->activate(result);
}

float neuron_cost(neuron_t *neuron, const train_sample_t *sample) {
	float result = 0.0;
	for (size_t i = 0; i < sample->len; i++) {
		expected_result_t *res = &sample->data[i];
		float prediction = neuron_predict(neuron, res->inputs);
		float delta = prediction - res->output;
		result += delta * delta / sample->len;
	}
	return result;
}

#define derivative_eps 1e-2
#define derivative(cost, argument, output) do { \
		float start = cost; \
		argument += derivative_eps; \
		float end = cost; \
		argument -= derivative_eps; \
		output = (end - start) / derivative_eps; \
	} while(0)

// TODO: make all using formula :)
neuron_cost_derivatives_t neuron_cost_derivatives(neuron_t *neuron, const train_sample_t *sample) {
	float bias_d;
	derivative(neuron_cost(neuron, sample), neuron->bias, bias_d);
	float *weights_d = malloc(sizeof(float) * neuron->size); // TODO: cache allocated memory or set by ptr	
	for (int i = 0; i < neuron->size; i++) {
		derivative(neuron_cost(neuron, sample), neuron->weights[i], weights_d[i]);
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
		printf(") = %f -> %f\n", result, res->output);
	}
}
