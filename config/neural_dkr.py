_hidden_size = 200
_code_len = 800
_save_dir_name = 'neural_dkr'
_save_suffix = '1'
_kp_save_path = 'save/%s/keyword_predictor_%s' % (_save_dir_name, _save_suffix)
_retrieval_save_path = 'save/%s/response_retrieval_%s' % (_save_dir_name, _save_suffix)
_log_save_path = 'save/%s/logs/training_logs.txt' % _save_dir_name
_conversation_save_path = 'save/%s/logs/conversation_logs.txt' % _save_dir_name
_simulation_save_path = 'save/%s/logs/simulation_logs.txt' % _save_dir_name
_max_epoch = 20
_dropout_rate = 0.5

_kp_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.005,
        }
    },
    "learning_rate_decay": {
        "type": "inverse_time_decay",
        "kwargs": {
            "decay_steps": 1600,
            "decay_rate": 0.8
        },
        "start_decay_step": 0,
        "end_decay_step": 16000,
    },
}

embedder_hparams = {
    "dim": 200,
    "dropout_rate": _dropout_rate,
    "dropout_strategy": 'element',
    "trainable": True,
    "initializer": {
        "type": "random_uniform_initializer",
        "kwargs": {
            "minval": -0.1,
            "maxval": 0.1,
            "seed": None
        }
    },
    "regularizer": {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0.
        }
    },
    "name": "word_embedder",
}

context_encoder_hparams = {
    "rnn_cell": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": _hidden_size,
        },
    }
}

_retrieval_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001,
        }
    },
    "learning_rate_decay": {
        "type": "inverse_time_decay",
        "kwargs": {
            "decay_steps": 1600,
            "decay_rate": 0.8
        },
        "start_decay_step": 0,
        "end_decay_step": 16000,
    },
}

source_encoder_hparams = {
    "encoder_minor_type": "BidirectionalRNNEncoder",
    "encoder_minor_hparams": {
        "rnn_cell_fw": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": _hidden_size,
            },
        },
        "rnn_cell_share_config": True
    },
    "encoder_major_type": "UnidirectionalRNNEncoder",
    "encoder_major_hparams": {
        "rnn_cell": {
            "type": "GRUCell",
            "kwargs": {
                "num_units": _hidden_size*2,
            },
        }
    }
}

target_encoder_hparams = {
    "rnn_cell_fw": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": _hidden_size,
        },
    },
    "rnn_cell_share_config": True
}

target_kwencoder_hparams = {
    "rnn_cell_fw": {
        "type": "GRUCell",
        "kwargs": {
            "num_units": _hidden_size,
        },
    },
    "rnn_cell_share_config": True
}
