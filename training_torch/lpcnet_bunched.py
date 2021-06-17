import numpy as np

import torch
from torch import nn


class MDense(nn.Module):
    def __init__(self, input_features, output_features):
        super(MDense, self).__init__()

        self.weight1 = nn.Parameter(torch.randn(output_features, input_features), requires_grad=True)
        nn.init.xavier_uniform_(self.weight1, gain=torch.nn.init.calculate_gain("tanh"))

        self.weight2 = nn.Parameter(torch.randn(output_features, input_features), requires_grad=True)
        nn.init.xavier_uniform_(self.weight2, gain=torch.nn.init.calculate_gain("tanh"))

        self.bias1 = nn.Parameter(torch.randn(output_features), requires_grad=True)
        self.bias2 = nn.Parameter(torch.randn(output_features), requires_grad=True)

        self.factor1 = nn.Parameter(torch.ones(output_features), requires_grad=True)
        self.factor2 = nn.Parameter(torch.ones(output_features), requires_grad=True)

    def forward(self, inputs):
        output1 = inputs.matmul(self.weight1.t()) + self.bias1
        output2 = inputs.matmul(self.weight2.t()) + self.bias2
        output1 = torch.tanh(output1) * self.factor1
        output2 = torch.tanh(output2) * self.factor2
        output = output1 + output2

        return output


class LPCNetModelBunch(nn.Module):
    def __init__(self, hparams):
        super(LPCNetModelBunch, self).__init__()

        self.n_samples_per_step = hparams.n_samples_per_step
        self.embedding_pitch_size = hparams.embedding_pitch_size
        self.embedding_size = hparams.embedding_size
        self.dense_feature_size = hparams.dense_feature_size
        self.ulaw = 2**hparams.ulaw
        self.rnn_units1 = hparams.rnn_units1
        self.rnn_units2 = hparams.rnn_units2
        self.frame_size = hparams.frame_size

        self.embed_pitch = nn.Embedding(hparams.pitch_max_period, self.embedding_pitch_size)
        self.embed_sig = nn.Embedding(self.ulaw, self.embedding_size)

        self.feature_conv1 = nn.Conv1d(self.embedding_pitch_size + hparams.nb_used_features,
                                       self.dense_feature_size, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.feature_conv1.weight, gain=torch.nn.init.calculate_gain("tanh"))

        self.feature_conv2 = nn.Conv1d(self.dense_feature_size, self.dense_feature_size, kernel_size=3)
        torch.nn.init.xavier_uniform_(self.feature_conv2.weight, gain=torch.nn.init.calculate_gain("tanh"))

        self.feature_dense1 = nn.Linear(self.dense_feature_size, self.dense_feature_size)
        torch.nn.init.xavier_uniform_(self.feature_dense1.weight, gain=torch.nn.init.calculate_gain("tanh"))

        self.feature_dense2 = nn.Linear(self.dense_feature_size, self.dense_feature_size)
        torch.nn.init.xavier_uniform_(self.feature_dense2.weight, gain=torch.nn.init.calculate_gain("tanh"))

        self.gru_a = nn.GRU(3*self.embedding_size*self.n_samples_per_step + self.dense_feature_size,
                            self.rnn_units1, batch_first=True)
        self.gru_b = nn.GRU(self.rnn_units1 + self.dense_feature_size,
                            self.rnn_units2, batch_first=True)

        self.md_1 = MDense(self.rnn_units2, self.ulaw)
        self.md_2 = MDense(self.rnn_units2 + self.embedding_size, self.ulaw)

        self.p_teacher_forcing = hparams.teacher_forcing

    def parse_decoder_inputs(self, cpcm_cexc):
        # [batch, 15*frame_size, 3*embedding_size] ==> [batch, 15*frame_size//r, 3*embedding_size*r]
        cpcm_cexc = cpcm_cexc.contiguous().view(cpcm_cexc.size(0), cpcm_cexc.size(1)//self.n_samples_per_step, -1)

        return cpcm_cexc

    def forward(self, in_data, features, periods, targets):
        """
        :param in_data: [batch, 15*frame_size, 3] (sig, pred, exc) shared embedding
        :param features: features: [batch, 15, nb_used_features]
        :param periods: periods: [batch, 15, 1]
        :param targets: [batch, 15*frame_size]
        :return: ulaw_probs: [batch, 15*frame_size, 2**ulaw]
        """

        ###################
        ##### Encoder #####
        ###################
        # [batch, 15, 38] + [batch, 15, embedding_pitch_size] ==> [batch, 15, 38 + embedding_pitch_size]
        pitch = self.embed_pitch(periods).squeeze(2)
        cat_feat = torch.cat((features, pitch), 2)

        # [batch, 15, 38 + embedding_pitch_size] ==> [batch, 15, embedding_size]
        cat_feat1 = cat_feat.permute(0, 2, 1)
        c_feat2 = torch.tanh(self.feature_conv1(cat_feat1))
        cfeat = torch.tanh(self.feature_conv2(c_feat2))
        c_feat2 = cfeat.permute(0, 2, 1)

        # [batch, 15, embedding_size] ==> [batch, 15, embedding_size]
        fdense1 = torch.tanh(self.feature_dense1(c_feat2))
        fdense2 = torch.tanh(self.feature_dense2(fdense1))

        # repeat features by self.frame_size//r times  ==>  [batch, 15*frame_size//r, dense_feature_size]
        repeat_tensor = in_data.new_ones(fdense2.size(1), dtype=torch.long) * self.frame_size//self.n_samples_per_step
        repeat_fdense2 = torch.repeat_interleave(fdense2, repeat_tensor, dim=1)

        ###################
        ##### Decoder #####
        ###################
        # [batch, 15*frame_size, 3] ==> [batch, 15*frame_size, 3*embedding_size]
        cpcm_exc = self.embed_sig(in_data)
        cpcm_exc = cpcm_exc.contiguous().view(cpcm_exc.size(0), cpcm_exc.size(1), -1)
        # [batch, 15*frame_size, 3*embedding_size] ==> [batch, 15*frame_size//r, 3*embedding_size*r]
        cpcm_exc = self.parse_decoder_inputs(cpcm_exc)

        # gru_a
        """
        [batch, 15*frame_size//r, 3*embed_size + dense_feature_size]  ==>  [batch, 15*frame_size//r, 384]
        """
        rnn_in = torch.cat((cpcm_exc, repeat_fdense2), 2)
        self.gru_a.flatten_parameters()
        gru_out1, _ = self.gru_a(rnn_in)

        # gru_b
        rnn_in2 = torch.cat((gru_out1, repeat_fdense2), 2)  # [batch, 15*frame_size//r, 384 + dense_feature_size]
        self.gru_b.flatten_parameters()
        gru_b_out, _ = self.gru_b(rnn_in2)  # [batch, 15*frame_size//r, rnn_units2]

        # results
        ulaw_probs = gru_b_out.new_zeros((targets.size(0), targets.size(1), self.ulaw))
        ulaw_probs[:, ::self.n_samples_per_step] = self.md_1(gru_b_out)
        context = gru_b_out
        threshold = np.random.uniform(0.0, 1.0)
        if threshold <= self.p_teacher_forcing:
            pred_exc = targets[:, 0::self.n_samples_per_step]
        else:
            pred_exc = torch.softmax(ulaw_probs[:, ::self.n_samples_per_step], dim=-1).argmax(-1)

        context = torch.cat((context, self.embed_sig(pred_exc)), dim=-1)
        ulaw_probs[:, 1::self.n_samples_per_step] = self.md_2(context)

        return ulaw_probs
