#pragma once

#include <malloc.h>
#include <stdarg.h>
#include <string.h>

#define define_array_type(type, name) \
	typedef struct { \
		size_t len; \
		type *data; \
	} name##_t; \
	\
	name##_t name##_of(const size_t count, ...); \
	void name##_pop(name##_t *arr); \
	void name##_free(const name##_t *arr);

#define impl_array_type(type, name) \
	name##_t name##_of(const size_t count, ...) { \
		type *data = malloc(sizeof(type) * count); \
		va_list args; \
		va_start(args, count); \
		for (size_t i = 0; i < count; i++) { \
			type value = va_arg(args, type); \
			memcpy(&data[i], &value, sizeof(type)); \
		} \
		va_end(args); \
		name##_t result = {count, data}; \
		return result; \
	} \
	void name##_pop(name##_t *arr) { arr->len--; } \

#define impl_array_simple_free(name) \
	void name##_free(const name##_t *arr) { \
		free(arr->data);\
	}

#define impl_array_nested_free(name, free_func) \
	void name##_free(const name##_t *arr) { \
		for (int i = 0; i < arr->len; i++) { \
			free_func(&arr->data[i]); \
		} \
		free(arr->data); \
	}

