#include <stdio.h>
#include "neuron.h"
#include "mathf.h"
#include "train.h"
#include "layer.h"

float b0[] = {0, 0};
float b1[] = {0, 1};
float b2[] = {1, 0};
float b3[] = {1, 1};

#define TRAIN_EPOCHS 1000000
#define TRAIN_RATE 1e-2

int main() {
	srand(100);

	// Training "AND Gate" perceptron
	train_sample_t and_sample = train_sample_of(4,
		expected_result(b0, 0),
		expected_result(b1, 0),
		expected_result(b2, 0),
		expected_result(b3, 1)	
	);
	train_sample_t or_sample = train_sample_of(4,
		expected_result(b0, 0),
		expected_result(b1, 1),
		expected_result(b2, 1),
		expected_result(b3, 1)
	);
	train_sample_t nand_sample = train_sample_of(4,	
		expected_result(b0, 1),
		expected_result(b1, 1),
		expected_result(b2, 1),
		expected_result(b3, 0)
	);

	neuron_t and = neuron_new_random(2, sigmoidf);
	neuron_t or = neuron_new_random(2, sigmoidf);
	neuron_t nand = neuron_new_random(2, sigmoidf);

	neuron_train_epochs(&and, &and_sample, TRAIN_RATE, TRAIN_EPOCHS);
	neuron_train_epochs(&or, &or_sample, TRAIN_RATE, TRAIN_EPOCHS);
	neuron_train_epochs(&nand, &nand_sample, TRAIN_RATE, TRAIN_EPOCHS);
	
	printf("----------- and  -----------\n");
	neuron_print_predictions(&and, &and_sample);
	printf("----------- or   -----------\n");
	neuron_print_predictions(&or, &or_sample);
	printf("----------- nand -----------\n");
	neuron_print_predictions(&nand, &nand_sample);

	// Wrapping the trained perceptron in the one-layer model
	printf("----------- xor  -----------\n");
	multilayer_model_t model = multilayer_model(
		model_layers_of(2,
			layer_of(2, 
				layer_neuron(or, layer_ids_of(2, 0, 1)),
				layer_neuron(nand, layer_ids_of(2, 0, 1))
			),
			layer_of(1,
				layer_neuron(and, layer_ids_of(2, 0, 1))
			)
		), 
		layer_ids_of(1, 0)
	);

	for (size_t a = 0; a < 2; a++) {
		for (size_t b = 0; b < 2; b++) {
			float data[] = {a, b};
			float *result = multilayer_model_predict(&model, data);
			printf("%d ^ %d -> %f\n", a, b, result[0]);
			free(result);
		}
	}	
	
	neuron_free(&or);
	neuron_free(&nand);
	neuron_free(&and);
	return 0;
}
