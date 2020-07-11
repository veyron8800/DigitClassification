#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 28
#define N_INPUT_2_1 28
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 26
#define OUT_WIDTH_2 26
#define N_FILT_2 4
#define OUT_HEIGHT_4 13
#define OUT_WIDTH_4 13
#define N_FILT_4 4
#define OUT_HEIGHT_5 11
#define OUT_WIDTH_5 11
#define N_FILT_5 8
#define OUT_HEIGHT_7 5
#define OUT_WIDTH_7 5
#define N_FILT_7 8
#define OUT_HEIGHT_8 3
#define OUT_WIDTH_8 3
#define N_FILT_8 16
#define N_LAYER_10 16
#define N_LAYER_12 10

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<16,6> layer12_t;
typedef ap_fixed<16,6> result_t;

#endif
