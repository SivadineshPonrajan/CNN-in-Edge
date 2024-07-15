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

#define FIXED_POINT	0	// Fixed point scaling factor, set to 0 when using floating point
#define NUMBER_MIN	-inf	// Max value for this numeric type
#define NUMBER_MAX	inf	// Min value for this numeric type
typedef float number_t;		// Standard size numeric type used for weights and activations
typedef float long_number_t;	// Long numeric type used for intermediate results

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


const float conv1d_bias[CONV_FILTERS] = {-0x1.648e2a0000000p-5, 0x1.c7c2e20000000p-4, 0x1.a24e8e0000000p-5, 0x1.09df600000000p-5, 0x1.d61ac00000000p-6, 0x1.edeac20000000p-5, 0x1.5d9fea0000000p-6, -0x1.1aeb4a0000000p-5, 0x1.09f67a0000000p-3, 0x1.3c391a0000000p-7}
;

const float conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{-0x1.a3ca7e0000000p-5, -0x1.5925240000000p-2, -0x1.2046240000000p-2}
, {-0x1.4dc0980000000p-6, 0x1.c0c9420000000p-5, -0x1.b0db760000000p-3}
, {0x1.fc17500000000p-3, 0x1.6546e20000000p-3, -0x1.65a66a0000000p-2}
}
, {{0x1.06e3dc0000000p-3, 0x1.d7b55c0000000p-2, 0x1.05da920000000p-3}
, {0x1.bd3dbc0000000p-3, -0x1.c237440000000p-2, 0x1.5920980000000p-6}
, {-0x1.058e060000000p-1, 0x1.8af6040000000p-4, -0x1.304cf80000000p-3}
}
, {{0x1.8823840000000p-3, -0x1.341cfc0000000p-2, 0x1.9483fa0000000p-3}
, {-0x1.0934160000000p-2, 0x1.661a0e0000000p-3, -0x1.370a200000000p-2}
, {0x1.9453c00000000p-2, 0x1.af67de0000000p-2, -0x1.b0ae140000000p-5}
}
, {{-0x1.5064d80000000p-2, 0x1.22aa4a0000000p-2, 0x1.1053ea0000000p-2}
, {-0x1.19452c0000000p-3, -0x1.3bc4380000000p-5, 0x1.884bb00000000p-3}
, {0x1.3ecb540000000p-2, 0x1.8a20bc0000000p-3, 0x1.84aa6a0000000p-3}
}
, {{-0x1.430fc20000000p-8, -0x1.2461e20000000p-2, -0x1.0c114a0000000p-3}
, {-0x1.03e91e0000000p-1, -0x1.a661760000000p-2, -0x1.9da11a0000000p-4}
, {-0x1.629bb60000000p-3, -0x1.acee860000000p-3, -0x1.1a84740000000p-2}
}
, {{0x1.5098f00000000p-3, 0x1.abe4540000000p-6, 0x1.4ec8440000000p-5}
, {0x1.0d6cd80000000p-3, 0x1.313fc00000000p-3, 0x1.8100140000000p-2}
, {0x1.4d8c9e0000000p-2, 0x1.138d940000000p-4, 0x1.b6d2760000000p-6}
}
, {{-0x1.48eb780000000p-2, 0x1.19599c0000000p-2, 0x1.a54aac0000000p-3}
, {-0x1.544cc60000000p-2, -0x1.ab34980000000p-3, -0x1.0b1d340000000p-2}
, {0x1.6fbe7c0000000p-7, 0x1.3a68a00000000p-4, 0x1.2a46aa0000000p-5}
}
, {{-0x1.6363060000000p-2, 0x1.5ac0420000000p-6, -0x1.23dd1e0000000p-2}
, {0x1.43f14a0000000p-2, 0x1.49f1fe0000000p-2, -0x1.11f3500000000p-2}
, {-0x1.33106e0000000p-3, 0x1.3d22ae0000000p-3, 0x1.32d4280000000p-2}
}
, {{-0x1.72207c0000000p-2, 0x1.41904a0000000p-3, -0x1.2550ea0000000p-2}
, {-0x1.0544ee0000000p-4, 0x1.118b6e0000000p-1, 0x1.0f84920000000p-2}
, {-0x1.1b95620000000p-2, -0x1.fc9de80000000p-18, -0x1.0a1b4c0000000p-2}
}
, {{0x1.7b12d60000000p-5, 0x1.2a68160000000p-2, 0x1.9842920000000p-4}
, {-0x1.45a95e0000000p-5, 0x1.30973a0000000p-2, -0x1.5d2b040000000p-7}
, {-0x1.1e53380000000p-2, -0x1.2bee3c0000000p-3, 0x1.98dde60000000p-4}
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


const float dense_bias[FC_UNITS] = {-0x1.d941d60000000p-5, -0x1.009b1e0000000p-4, -0x1.3202840000000p-4, 0x1.fcace20000000p-5, -0x1.0ecffa0000000p-4, 0x1.f661740000000p-5}
;

const float dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{0x1.544d4a0000000p-3, 0x1.1c30b80000000p-2, -0x1.f167ca0000000p-8, 0x1.b717d20000000p-4, 0x1.8d81820000000p-2, -0x1.317ef60000000p-3, 0x1.2d4d600000000p-2, 0x1.a839de0000000p-8, 0x1.b29da60000000p-4, -0x1.f61a360000000p-7, 0x1.7a13f60000000p-6, -0x1.0081220000000p-4, 0x1.17aca40000000p-3, -0x1.cb229a0000000p-5, -0x1.dcc7020000000p-3, 0x1.d443220000000p-3, -0x1.38682e0000000p-3, -0x1.0f8f2c0000000p-4, -0x1.57ede60000000p-3, 0x1.5a63a40000000p-3, -0x1.44ea5a0000000p-3, 0x1.6dd0920000000p-6, -0x1.150a580000000p-2, -0x1.6f90720000000p-3, -0x1.86d78e0000000p-2, 0x1.6a90c80000000p-3, -0x1.42768e0000000p-4, 0x1.6db9ea0000000p-4, -0x1.1a4c2a0000000p-2, -0x1.ea538e0000000p-6, -0x1.99214a0000000p-3, -0x1.3835a20000000p-3, 0x1.5e446e0000000p-2, 0x1.5e7ec60000000p-2, -0x1.970a0e0000000p-4, 0x1.40da6a0000000p-4, -0x1.de5a780000000p-4, -0x1.d078340000000p-3, 0x1.a3e45c0000000p-3, -0x1.32da5c0000000p-4, 0x1.c034f60000000p-4, -0x1.325bca0000000p-2, -0x1.efafd80000000p-4, 0x1.1c6bc00000000p-3, 0x1.f7c5740000000p-5, -0x1.33eb2c0000000p-3, 0x1.51a2e00000000p-2, -0x1.9898120000000p-4, 0x1.1c20fc0000000p-2, -0x1.1610100000000p-7}
, {0x1.03b4340000000p-2, 0x1.39332e0000000p-3, 0x1.4e83a00000000p-2, 0x1.b8401c0000000p-3, 0x1.aa35b20000000p-3, 0x1.8ebee20000000p-3, 0x1.a4d4f80000000p-3, 0x1.ed3f6e0000000p-2, 0x1.eb5de00000000p-2, 0x1.85adcc0000000p-3, -0x1.1895080000000p-2, 0x1.56967e0000000p-3, 0x1.1e36ea0000000p-3, -0x1.ceef940000000p-8, 0x1.72a3400000000p-3, 0x1.533c280000000p-5, 0x1.21c8ca0000000p-7, -0x1.c926180000000p-4, -0x1.e849840000000p-3, 0x1.25cc940000000p-4, -0x1.6b15300000000p-2, -0x1.cd97d80000000p-3, -0x1.9bd6b60000000p-3, -0x1.34b89e0000000p-2, -0x1.c192380000000p-3, -0x1.5725920000000p-4, -0x1.b2f2240000000p-2, -0x1.21ba6c0000000p-2, -0x1.c9d0fa0000000p-2, -0x1.3afa520000000p-3, -0x1.0f80ba0000000p-2, 0x1.7baae00000000p-3, -0x1.82ead60000000p-6, 0x1.9880d00000000p-3, 0x1.27e7fe0000000p-4, -0x1.040aae0000000p-3, 0x1.892e340000000p-3, -0x1.e824a40000000p-4, 0x1.e1643e0000000p-3, 0x1.1836820000000p-3, 0x1.03806e0000000p-6, -0x1.b17f080000000p-3, -0x1.541e320000000p-3, -0x1.06ca160000000p-6, -0x1.70dbae0000000p-3, 0x1.387d680000000p-2, 0x1.ed8ba20000000p-9, -0x1.5e929a0000000p-6, -0x1.73e01c0000000p-4, -0x1.dc54dc0000000p-6}
, {0x1.41ca580000000p-2, 0x1.2c1f0e0000000p-2, 0x1.37c8440000000p-6, -0x1.7f076a0000000p-4, 0x1.52c7e40000000p-4, -0x1.4367480000000p-4, 0x1.e040ba0000000p-2, 0x1.e81a6e0000000p-3, 0x1.a1b1020000000p-3, 0x1.d570020000000p-2, -0x1.b3c42c0000000p-3, -0x1.f3086a0000000p-3, -0x1.1e77ac0000000p-2, 0x1.f0bc760000000p-4, 0x1.3c02d20000000p-5, -0x1.80d8600000000p-2, 0x1.dab2f40000000p-3, -0x1.1573260000000p-3, -0x1.e853000000000p-3, -0x1.e073020000000p-8, -0x1.126a660000000p-2, -0x1.9e93100000000p-2, -0x1.1fcdfa0000000p-2, 0x1.faade00000000p-5, 0x1.59aa480000000p-5, -0x1.2148240000000p-3, -0x1.cef00c0000000p-3, -0x1.0014ca0000000p-2, -0x1.574d520000000p-2, 0x1.53951c0000000p-4, -0x1.b232920000000p-3, -0x1.5b56fe0000000p-4, -0x1.a911e60000000p-3, 0x1.5874720000000p-2, -0x1.6695d60000000p-4, -0x1.fa52a40000000p-4, 0x1.ffb20a0000000p-3, -0x1.da367e0000000p-3, -0x1.e75c660000000p-7, 0x1.ec5e7e0000000p-3, -0x1.cc0e920000000p-2, 0x1.10fcbc0000000p-6, -0x1.224efa0000000p-2, -0x1.57ed560000000p-3, -0x1.dfcb1a0000000p-2, 0x1.a94fe00000000p-3, 0x1.0daba20000000p-2, 0x1.8c2eac0000000p-3, -0x1.37a0620000000p-3, 0x1.9075040000000p-7}
, {-0x1.9e4d9c0000000p-3, -0x1.5b251a0000000p-3, 0x1.1c36940000000p-3, -0x1.10c8080000000p-4, 0x1.e45bf20000000p-6, 0x1.6282100000000p-5, -0x1.a636dc0000000p-4, -0x1.7ce0ce0000000p-3, 0x1.12238a0000000p-5, -0x1.658d300000000p-4, 0x1.5414740000000p-4, -0x1.c596580000000p-3, -0x1.04fb860000000p-2, -0x1.b7279a0000000p-3, 0x1.0211000000000p-2, -0x1.c18a2c0000000p-6, 0x1.9db7620000000p-4, 0x1.66c4d00000000p-5, 0x1.04d2500000000p-4, 0x1.52823a0000000p-3, 0x1.9cf51c0000000p-2, -0x1.9c59760000000p-3, 0x1.5536680000000p-4, 0x1.a1afd80000000p-3, 0x1.be2bdc0000000p-4, 0x1.83083e0000000p-2, 0x1.4e52f40000000p-2, 0x1.50ce440000000p-2, 0x1.7ee7dc0000000p-6, 0x1.2e5d240000000p-2, -0x1.077f420000000p-4, 0x1.f8adbc0000000p-3, -0x1.1784a40000000p-3, -0x1.4a5cc60000000p-7, 0x1.bb6fd60000000p-6, -0x1.ab8dfc0000000p-4, 0x1.ce36b40000000p-6, 0x1.fe19da0000000p-8, -0x1.4f6dba0000000p-5, 0x1.23700c0000000p-3, 0x1.b2ad660000000p-3, 0x1.2f1bea0000000p-2, -0x1.21bd520000000p-6, 0x1.341f920000000p-4, 0x1.46baa20000000p-2, -0x1.c28f120000000p-3, 0x1.61c2060000000p-4, -0x1.af1e4e0000000p-3, 0x1.63e5680000000p-4, -0x1.6b5a4e0000000p-2}
, {0x1.9296340000000p-3, -0x1.45e7fa0000000p-4, -0x1.d9a6a00000000p-5, -0x1.e3c4e60000000p-6, -0x1.e7f26e0000000p-4, 0x1.c6b65c0000000p-5, -0x1.2683960000000p-5, -0x1.5e98ac0000000p-5, 0x1.328dc40000000p-4, 0x1.84d0c80000000p-2, 0x1.be14d00000000p-3, 0x1.f0bdcc0000000p-4, -0x1.a7fc860000000p-5, 0x1.4e09c00000000p-2, -0x1.eb33580000000p-3, 0x1.11741c0000000p-4, 0x1.37e8f00000000p-2, -0x1.86f9fa0000000p-3, 0x1.5411b20000000p-4, -0x1.be73600000000p-5, -0x1.be721e0000000p-2, -0x1.d138be0000000p-2, -0x1.484c4c0000000p-2, -0x1.58ce140000000p-3, -0x1.7d9a1e0000000p-2, 0x1.06906c0000000p-3, -0x1.861d260000000p-5, -0x1.ba12160000000p-2, -0x1.95ff2e0000000p-8, -0x1.3e1ff40000000p-3, 0x1.6c3ec20000000p-2, 0x1.201b920000000p-2, 0x1.8072240000000p-2, 0x1.01e0180000000p-3, -0x1.b1e8780000000p-4, -0x1.bf8dce0000000p-3, 0x1.07bf3e0000000p-3, 0x1.7ec8000000000p-3, -0x1.d500240000000p-3, -0x1.14bf440000000p-2, -0x1.1050340000000p-2, -0x1.2ab25e0000000p-5, -0x1.27eb640000000p-4, 0x1.1344400000000p-8, -0x1.664f880000000p-3, 0x1.4e84060000000p-2, 0x1.a94d160000000p-3, -0x1.ca7f0e0000000p-3, -0x1.76e9b80000000p-3, -0x1.e6f6e00000000p-6}
, {-0x1.5b56aa0000000p-2, 0x1.3bd5440000000p-4, 0x1.3b23460000000p-3, 0x1.fbbcc20000000p-3, -0x1.8716cc0000000p-2, -0x1.dbb9360000000p-2, -0x1.c9e60c0000000p-2, -0x1.07c1d60000000p-3, -0x1.dc35180000000p-2, 0x1.737fe80000000p-9, 0x1.2736720000000p-2, -0x1.de11d60000000p-6, -0x1.0dd9a80000000p-2, 0x1.42df3a0000000p-2, 0x1.64aaea0000000p-3, 0x1.17a4c80000000p-5, -0x1.06a37e0000000p-2, -0x1.3141860000000p-3, 0x1.a7154e0000000p-4, -0x1.0742220000000p-2, -0x1.9bbef60000000p-6, 0x1.70f4360000000p-2, 0x1.110de00000000p-5, 0x1.64156c0000000p-2, 0x1.40663e0000000p-3, -0x1.7225220000000p-3, 0x1.99feee0000000p-5, 0x1.08f5a20000000p-2, 0x1.0821280000000p-8, 0x1.0cb3380000000p-2, 0x1.49bfec0000000p-4, 0x1.ddc4980000000p-5, 0x1.f8219a0000000p-5, -0x1.0ede9e0000000p-2, 0x1.099cee0000000p-2, -0x1.3d56a40000000p-3, 0x1.9876560000000p-3, -0x1.0e99b60000000p-2, 0x1.09597c0000000p-2, -0x1.c6c6120000000p-4, 0x1.bd706e0000000p-2, -0x1.3b4e740000000p-3, 0x1.8f83bc0000000p-2, 0x1.cdd4400000000p-4, 0x1.3e16de0000000p-4, -0x1.33fd160000000p-2, -0x1.6f53f80000000p-3, 0x1.e2fe000000000p-3, 0x1.7670600000000p-3, 0x1.3cbf640000000p-5}
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


const float dense_1_bias[FC_UNITS] = {0x1.dc6d9e0000000p-5, -0x1.dc6d9c0000000p-5}
;

const float dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{-0x1.ffcfda0000000p-3, -0x1.9525f40000000p-1, -0x1.d9267c0000000p-1, 0x1.df852c0000000p-2, -0x1.c2834a0000000p-7, 0x1.0482340000000p-4}
, {0x1.93794e0000000p-1, 0x1.62c4c20000000p-1, -0x1.e9b5360000000p-2, -0x1.bd32340000000p-1, 0x1.f8a1f20000000p-2, -0x1.ef89820000000p-1}
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
