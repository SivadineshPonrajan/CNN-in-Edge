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


const int16_t conv1d_bias[CONV_FILTERS] = {21, -37, 0, -24, 65, 38, -29, -24, -9, 68}
;

const int16_t conv1d_kernel[CONV_FILTERS][INPUT_CHANNELS][CONV_KERNEL_SIZE] = {{{109, 89, 47}
, {83, 285, 41}
, {-74, 247, 246}
}
, {{206, 214, 255}
, {79, -70, -31}
, {-39, -123, 46}
}
, {{-136, 179, 191}
, {66, 40, -129}
, {-84, -47, 196}
}
, {{-225, -72, 273}
, {-94, 162, -53}
, {83, -86, 40}
}
, {{72, 46, -36}
, {136, -173, -161}
, {196, -157, 59}
}
, {{-61, 48, 54}
, {67, -50, 127}
, {65, 78, 1}
}
, {{278, 7, 16}
, {122, -73, 57}
, {-187, -62, 52}
}
, {{267, -28, -48}
, {-11, 11, 102}
, {-186, 25, -28}
}
, {{-39, -26, -84}
, {110, 59, 9}
, {70, -154, -3}
}
, {{8, -146, 107}
, {271, 157, 269}
, {-84, -99, 90}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE