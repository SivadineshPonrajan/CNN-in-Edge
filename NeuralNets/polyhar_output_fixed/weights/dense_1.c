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


const int16_t dense_1_bias[FC_UNITS] = {45, -45}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{413, 371, 215, -37, 319, -409}
, {253, 153, 474, -424, 517, 240}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS