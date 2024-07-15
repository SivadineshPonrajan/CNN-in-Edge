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