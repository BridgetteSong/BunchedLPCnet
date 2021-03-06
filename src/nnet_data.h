/*This file is automatically generated from a Pytorch model*/

#ifndef RNN_DATA_H
#define RNN_DATA_H

#include "nnet.h"

#define GRU_A_EMBED_SIG_1_OUT_SIZE 1152
extern const EmbeddingLayer gru_a_embed_sig_1;

#define GRU_A_EMBED_PRED_1_OUT_SIZE 1152
extern const EmbeddingLayer gru_a_embed_pred_1;

#define GRU_A_EMBED_EXC_1_OUT_SIZE 1152
extern const EmbeddingLayer gru_a_embed_exc_1;

#define GRU_A_EMBED_SIG_0_OUT_SIZE 1152
extern const EmbeddingLayer gru_a_embed_sig_0;

#define GRU_A_EMBED_PRED_0_OUT_SIZE 1152
extern const EmbeddingLayer gru_a_embed_pred_0;

#define GRU_A_EMBED_EXC_0_OUT_SIZE 1152
extern const EmbeddingLayer gru_a_embed_exc_0;

#define GRU_A_DENSE_FEATURE_OUT_SIZE 1152
extern const DenseLayer gru_a_dense_feature;

#define EMBED_PITCH_OUT_SIZE 74
extern const EmbeddingLayer embed_pitch;

#define FEATURE_CONV1_OUT_SIZE 128
#define FEATURE_CONV1_STATE_SIZE (112*2)
#define FEATURE_CONV1_DELAY 1
extern const Conv1DLayer feature_conv1;

#define FEATURE_CONV2_OUT_SIZE 128
#define FEATURE_CONV2_STATE_SIZE (128*2)
#define FEATURE_CONV2_DELAY 1
extern const Conv1DLayer feature_conv2;

#define FEATURE_DENSE1_OUT_SIZE 128
extern const DenseLayer feature_dense1;

#define FEATURE_DENSE2_OUT_SIZE 128
extern const DenseLayer feature_dense2;

#define GRU_A_OUT_SIZE 384
#define GRU_A_STATE_SIZE 384
extern const GRULayer gru_a;

#define GRU_B_OUT_SIZE 16
#define GRU_B_STATE_SIZE 16
extern const GRULayer gru_b;

#define DUAL_FC_1_OUT_SIZE 256
extern const MDenseLayer dual_fc_1;

#define EMBED_SIG_OUT_SIZE 128
extern const EmbeddingLayer embed_sig;

#define DUAL_FC_2_OUT_SIZE 256
extern const MDenseLayer dual_fc_2;

#define SPARSE_GRU_A_OUT_SIZE 384
#define SPARSE_GRU_A_STATE_SIZE 384
extern const SparseGRULayer sparse_gru_a;

#define MD_EMBED_SIG_OUT_SIZE 512
extern const EmbeddingLayer md_embed_sig;

#define MAX_RNN_NEURONS 384

#define MAX_CONV_INPUTS 384

#define MAX_MDENSE_TMP 512

typedef struct {
  float feature_conv1_state[FEATURE_CONV1_STATE_SIZE];
  float feature_conv2_state[FEATURE_CONV2_STATE_SIZE];
  float gru_a_state[GRU_A_STATE_SIZE];
  float gru_b_state[GRU_B_STATE_SIZE];
} NNetState;


#endif
