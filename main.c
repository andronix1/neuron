#include <stdio.h>
#include "neuron.h"
#include "mathf.h"
#include "train.h"

/*
typedef struct {
	float *inputs;
	float output;
} expected_result_t;
*/

float dx0[] = {0, 0};
float dx1[] = {0, 1};
float dx2[] = {1, 0};
float dx3[] = {1, 1};
expected_result_t results[] = {
	{.inputs = dx0, .output = 0},
	{.inputs = dx1, .output = 1},
	{.inputs = dx2, .output = 1},
	{.inputs = dx3, .output = 1},
};
#define DATA_LENGTH (sizeof(results) / sizeof(expected_result_t))

#define TRAIN_EPOCHS 1000000
#define PRINT_EPOCHS 10

int main() {
	srand(100);
	train_sample_t sample = {
		.results = results,
		.length = DATA_LENGTH
	};

	neuron_t neuron;
        neuron_init(&neuron, 2, sigmoidf);
	neuron_random(&neuron);
	
	for (size_t i = 0; i < TRAIN_EPOCHS; i++) {
		neuron_train_epoch(&neuron, &sample, 1e-1);
		float cost = neuron_cost(&neuron, &sample);
		if (i % (TRAIN_EPOCHS / PRINT_EPOCHS) == 0) {
			printf("epoch %d: cost = %f\n", i, cost);
		}
	}

	neuron_print_predictions(&neuron, &sample);

	neuron_free(&neuron);
	return 0;
}
