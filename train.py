import tensorflow as tf
import importlib
import os

if __name__ == '__main__':
    flags = tf.flags
    # flags.DEFINE_string('data', 'data_config', 'The data config')
    flags.DEFINE_string('dataset', 'TGPC', 'The dataset, supports TGPC / CWC.')
    flags.DEFINE_string('agent', 'neural_dkr', 'The agent type, \
        supports neural_dkr / kernel / matrix / neural / retrieval / retrieval_stgy.')
    flags.DEFINE_string('mode', 'train_kw', 'The mode, supports train_kw / test_kw / train / test')
    FLAGS = flags.FLAGS

    # Target-Guided PersonaChat Dataset
    if FLAGS.dataset == 'TGPC':
        config_dir = 'config.'
        save_dir = 'save/'
        os.environ['is_weibo'] = 'False'
    # Chinese Weibo Conversation Dataset
    elif FLAGS.dataset == 'CWC':
        config_dir = 'config_weibo.'
        save_dir = 'save_weibo/'
        os.environ['is_weibo'] = 'True'

    config_data = importlib.import_module(config_dir + 'data_config')
    config_model = importlib.import_module(config_dir + FLAGS.agent)
    model = importlib.import_module('model.' + FLAGS.agent)
    predictor = model.Predictor(config_model, config_data, FLAGS.mode)
    if not os.path.exists(save_dir + FLAGS.agent):
        os.makedirs(save_dir + FLAGS.agent)

    if FLAGS.mode == 'train_kw':
        predictor.train_keywords()
        predictor.test_keywords()
    if FLAGS.mode == 'test_kw':
        predictor.test_keywords()
    if FLAGS.mode == 'train':
        predictor.train()
        predictor.test()
    if FLAGS.mode == 'test':
        predictor.test()
