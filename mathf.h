#pragma once

#include <stdlib.h>
#include <math.h>

typedef float (*func_t)(float);

float randf(void);
float sigmoidf(float x);
float sigmoidf_derivative(float x);
