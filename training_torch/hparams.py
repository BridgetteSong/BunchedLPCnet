import tensorflow as tf


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=120,
        frame_size=160,
        bfcc_band=18,
        nb_features=55, #nb_features = bfcc_band * 2 + 3 + 16,
        nb_used_features=38, #2*bfcc_band + pitch + pitch corr
        ulaw=8,
        pitch_idx=36, # pitch_idx = 2 * bfcc_band
        pitch_max_period=256, #768 for 48K, 256 for 16K
        embedding_size=128,
        embedding_pitch_size=74, # make (embedding_pitch_size+nb_used_features)%16==0 to accelerate  
        dense_feature_size=128,
        rnn_units1=384,
        rnn_units2=16,
        n_samples_per_step=2,
        spartify=True,
        teacher_forcing=1.0, 

        ################################
        # Data Parameters             #
        ################################
        features="../data/input.f32",
        pcms="../data/input.u8",
        checkpoint_path='checkpoint/outdir',
        log_dir='checkpoint/outdir/logs',

        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate=1e-3,
        lr_decay=5e-6,
        grad_clip_thresh=1.0,
        batch_size=64,
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
