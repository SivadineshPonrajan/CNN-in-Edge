#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>

#define FIXED_POINT	9	// Fixed point scaling factor, set to 0 when using floating point
#define NUMBER_MIN	-32768	// Max value for this numeric type
#define NUMBER_MAX	32767	// Min value for this numeric type
typedef int16_t number_t;		// Standard size numeric type used for weights and activations
typedef int32_t long_number_t;	// Long numeric type used for intermediate results

#ifndef min
static inline long_number_t min(long_number_t a, long_number_t b) {
	if (a <= b)
		return a;
	return b;
}
#endif

#ifndef max
static inline long_number_t max(long_number_t a, long_number_t b) {
	if (a >= b)
		return a;
	return b;
}
#endif

#if FIXED_POINT > 0 // Scaling/clamping for fixed-point representation
static inline long_number_t scale_number_t(long_number_t number) {
	return number >> FIXED_POINT;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) max(NUMBER_MIN, min(NUMBER_MAX, number));
}
#else // No scaling/clamping required for floating-point
static inline long_number_t scale_number_t(long_number_t number) {
	return number;
}
static inline number_t clamp_to_number_t(long_number_t number) {
	return (number_t) number;
}
#endif


#endif //__NUMBER_H__
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      3
#define INPUT_SAMPLES       32
#define CONV_FILTERS        10
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

typedef number_t conv1d_output_type[CONV_FILTERS][CONV_OUTSAMPLES];

static inline void conv1d(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES],               // IN
  const number_t kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE], // IN

  const number_t bias[CONV_FILTERS],						                // IN

  number_t output[CONV_FILTERS][CONV_OUTSAMPLES]) {               // OUT

  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  short input_x;
  long_number_t	kernel_mac;
  static long_number_t	output_acc[CONV_OUTSAMPLES];
  long_number_t tmp;

  for (k = 0; k < CONV_FILTERS; k++) { 
    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
      output_acc[pos_x] = 0;
	    for (z = 0; z < INPUT_CHANNELS; z++) {

        kernel_mac = 0; 
        for (x = 0; x < CONV_KERNEL_SIZE; x++) {
          input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;
          if (input_x < 0 || input_x >= INPUT_SAMPLES) // ZeroPadding1D
            tmp = 0;
          else
            tmp = input[z][input_x] * kernel[k][z][x]; 
          kernel_mac = kernel_mac + tmp; 
        }

	      output_acc[pos_x] = output_acc[pos_x] + kernel_mac; 
      }
      output_acc[pos_x] = scale_number_t(output_acc[pos_x]);

      output_acc[pos_x] = output_acc[pos_x] + bias[k]; 

    }

    for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) {
#ifdef ACTIVATION_LINEAR
      output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc[pos_x] < 0)
        output[k][pos_x] = 0;
      else
        output[k][pos_x] = clamp_to_number_t(output_acc[pos_x]);
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
/**
  ******************************************************************************
  * @file    weights/conv.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_CHANNELS    3
#define CONV_FILTERS      10
#define CONV_KERNEL_SIZE  3


const int16_t conv1d_bias[CONV_FILTERS] = {-23, -15, 12, -19, -45, -8, 77, -11, 6, -20}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-164, -170, 105}
, {-47, -8, 107}
, {23, -88, 113}
}
, {{-44, 163, -14}
, {-80, 206, 224}
, {47, 122, 138}
}
, {{-117, 0, 90}
, {-282, 56, -165}
, {-37, 150, 153}
}
, {{147, -63, 65}
, {137, 118, 162}
, {139, 153, -122}
}
, {{-222, -128, -132}
, {-65, -86, 101}
, {143, 80, 24}
}
, {{23, 52, -100}
, {192, -5, 16}
, {-55, 57, 46}
}
, {{29, -91, -175}
, {200, 229, 246}
, {111, -227, -124}
}
, {{190, -10, -129}
, {-1, 72, -55}
, {-190, 100, 141}
}
, {{199, 274, -95}
, {30, 134, -93}
, {36, -175, -80}
}
, {{172, 73, -50}
, {-95, 53, -215}
, {137, -59, -81}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  10
#define INPUT_SAMPLES   30
#define POOL_SIZE       6
#define POOL_STRIDE     6
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

typedef number_t max_pooling1d_output_type[INPUT_CHANNELS][POOL_LENGTH];

static inline void max_pooling1d(
  const number_t input[INPUT_CHANNELS][INPUT_SAMPLES], 	    // IN
  number_t output[INPUT_CHANNELS][POOL_LENGTH]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  number_t max, tmp; 

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
      max = input[k][pos_x*POOL_STRIDE];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max = 0;
      x = 0;
#endif
      for (; x < POOL_SIZE; x++) {
        tmp = input[k][(pos_x*POOL_STRIDE)+x]; 
        if (max < tmp)
          max = tmp;
      }
      output[k][pos_x] = max; 
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_DIM [5][10]
#define OUTPUT_DIM 50

//typedef number_t *flatten_output_type;
typedef number_t flatten_output_type[OUTPUT_DIM];

#define flatten //noop (IN, OUT)  OUT = (number_t*)IN

#undef INPUT_DIM
#undef OUTPUT_DIM

/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 50
#define FC_UNITS 6
#define ACTIVATION_LINEAR

typedef number_t dense_output_type[FC_UNITS];

static inline void dense(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 50
#define FC_UNITS 6


const int16_t dense_bias[FC_UNITS] = {18, -20, -22, 23, 20, 19}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{111, -157, -156, -11, -34, 94, 95, 134, -133, -13, 16, -59, 151, -10, -139, 122, 90, -16, 112, -108, 38, -76, 40, -106, -95, -39, 131, 15, 35, 43, 39, 244, 13, -41, 77, 57, -23, 43, -72, -49, -75, 9, -152, 80, -131, 74, 37, 62, -29, -85}
, {-52, 124, -28, 88, 128, -112, -100, -61, -140, -20, -16, -156, -114, -70, -97, -80, -89, -12, 36, 1, 32, 40, 45, -27, 86, 71, -97, -32, -121, 100, -81, -75, -55, -24, -221, 168, -18, -98, 127, -97, 216, 25, 195, 109, 127, -148, -58, 21, -98, -135}
, {180, 68, 140, -69, 28, 94, -101, -50, 81, -75, -103, -52, 94, -114, -44, 43, -45, 135, -49, 40, 10, -38, -43, 146, 59, -132, 146, -71, 58, -52, -140, 33, -36, -176, -134, 134, 56, -96, 193, 170, -71, 114, 109, 84, 230, 51, 27, -61, 142, 127}
, {99, -128, 18, 99, 132, -28, 2, -82, -30, -123, -37, 86, -167, 33, -96, -71, 60, -93, 11, -66, -15, 132, -30, 56, -174, -109, -134, 17, 61, -111, -58, 105, 209, 237, 54, -195, -95, 4, -146, -83, 103, -174, -182, -27, 51, -76, -134, 147, 131, -156}
, {-52, -103, 116, -131, -14, 7, -128, 144, 134, -4, 97, 32, 147, 97, -51, 57, 54, 15, 179, -18, -66, -104, -26, 37, 3, 20, 111, -156, 143, 157, 181, 241, 16, -49, 89, -186, 2, 118, -157, -101, -51, -193, -181, -55, 40, 142, 68, -131, -102, 88}
, {-136, -88, -165, 129, 118, -63, -42, 118, -66, 69, -77, 96, -169, 24, 153, 55, 108, 138, 93, -95, -75, 43, -134, 130, 58, -128, 99, -34, 154, 157, 67, 46, -59, 0, 43, -141, 59, 18, 79, -163, -88, 10, -131, -25, 14, -3, -187, -114, 143, -121}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_SAMPLES 6
#define FC_UNITS 2
#define ACTIVATION_LINEAR

typedef number_t dense_1_output_type[FC_UNITS];

static inline void dense_1(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]) {			                // OUT

  unsigned short k, z; 
  long_number_t output_acc; 

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0; 
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ( kernel[k][z] * input[z] ); 

    output_acc = scale_number_t(output_acc);

    output_acc = output_acc + bias[k]; 


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = clamp_to_number_t(output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0)
      output[k] = 0;
    else
      output[k] = clamp_to_number_t(output_acc);
#endif
  }
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#define INPUT_SAMPLES 6
#define FC_UNITS 2


const int16_t dense_1_bias[FC_UNITS] = {19, -19}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{421, -243, -332, 296, 205, 50}
, {-420, 463, 171, 78, -361, -337}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define MODEL_OUTPUT_SAMPLES 2
#define MODEL_INPUT_SAMPLES 32 // node 0 is InputLayer so use its output shape as input shape of the model
#define MODEL_INPUT_CHANNELS 3

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  //dense_1_output_type dense_1_output);
  number_t output[MODEL_OUTPUT_SAMPLES]);

#endif//__MODEL_H__
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"

 // InputLayer is excluded
#include "conv1d.c"
#include "weights/conv1d.c" // InputLayer is excluded
#include "max_pooling1d.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c" // InputLayer is excluded
#include "dense_1.c"
#include "weights/dense_1.c"
#endif

void cnn(
  const number_t input[MODEL_INPUT_CHANNELS][MODEL_INPUT_SAMPLES],
  dense_1_output_type dense_1_output) {

  // Output array allocation
  static union {
    conv1d_output_type conv1d_output;
    dense_output_type dense_output;
  } activations1;

  static union {
    max_pooling1d_output_type max_pooling1d_output;
    flatten_output_type flatten_output;
  } activations2;


  //static union {
//
//    static input_1_output_type input_1_output;
//
//    static conv1d_output_type conv1d_output;
//
//    static max_pooling1d_output_type max_pooling1d_output;
//
//    static flatten_output_type flatten_output;
//
//    static dense_output_type dense_output;
//
  //} activations;

  // Model layers call chain
 // InputLayer is excluded 
  conv1d(
     // First layer uses input passed as model parameter
    input,
    conv1d_kernel,
    conv1d_bias,
    activations1.conv1d_output
  );
 // InputLayer is excluded 
  max_pooling1d(
    
    activations1.conv1d_output,
    activations2.max_pooling1d_output
  );
 // InputLayer is excluded 
  flatten(
    
    activations2.max_pooling1d_output,
    activations2.flatten_output
  );
 // InputLayer is excluded 
  dense(
    
    activations2.flatten_output,
    dense_kernel,
    dense_bias,
    activations1.dense_output
  );
 // InputLayer is excluded 
  dense_1(
    
    activations1.dense_output,
    dense_1_kernel,
    dense_1_bias, // Last layer uses output passed as model parameter
    dense_1_output
  );

}
