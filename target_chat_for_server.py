import tensorflow as tf
import importlib
import random
import os
from preprocess.data_utils import utter_preprocess, is_reach_goal
from utils.log_utils import create_logs, add_log
import time

class Target_Chat:
    def __init__(self, model, config_model, config_data):
        g = tf.Graph()
        with g.as_default():
            self.agent = model.Predictor(config_model, config_data, 'test')
            self.sess = tf.Session(graph=g, config=self.agent.gpu_config)
            self.agent.retrieve_init(self.sess)

        self.target_set = config_data._target_keywords_for_simulation
        self.start_corpus = config_data._start_corpus
        self.max_turns = config_data._max_turns
        self.conversation_save_path = config_model._conversation_save_path
        self.current_sessions = 0

        create_logs(self.conversation_save_path)

    def chat(self, user_history=[]):
        print(user_history)
        responses = []
        # if is the beginning of a conversation
        if len(user_history) == 0:
            self._reset()
            reply = self.start_utterance
            add_log(self.conversation_save_path, '-------- Session {} --------'.format(self.current_sessions))
            add_log(self.conversation_save_path, 'START: {}'.format(reply))
        else:
            user_input = user_history[-1]
            source = utter_preprocess(user_history, self.agent.data_config._max_seq_len)
            reply = self.agent.retrieve(source, self.sess)
            add_log(self.conversation_save_path, 'HUMAN: {}'.format(user_input), print_details=False)
            add_log(self.conversation_save_path, 'AGENT: {}'.format(reply))
        responses.append(reply)
        self.current_turns += 1

        # if the last two utterances contain target keyword
        if is_reach_goal(' '.join(user_history[-2:]), self.target_keyword):
            end_message = '[SUCCESS] target: \'{}\'.'.format(self.target_keyword)
            add_log(self.conversation_save_path, end_message)
            responses.append(end_message)
        # if is out of the max dialogue turn
        elif self.current_turns > self.max_turns:
            end_message = '[FAIL] out of the max dialogue turns, target: \'{}\'.'.format(self.target_keyword)
            add_log(self.conversation_save_path, end_message)
            responses.append(end_message)

        return responses

    def _reset(self):
        self.current_turns = 0
        self.current_sessions += 1
        self.start_utterance = random.sample(self.start_corpus, 1)[0]
        self.target_keyword = random.sample(self.target_set,1)[0]
        self.agent.target = self.target_keyword
        self.agent.score = 0.
        self.agent.reply_list = []

def init_target_chat(agent_name, dataset):
    # Target-Guided PersonaChat Dataset
    if dataset == 'TGPC':
        config_dir = 'config.'
        os.environ['is_weibo'] = 'False'
    # Chinese Weibo Conversation Dataset
    elif dataset == 'CWC':
        config_dir = 'config_weibo.'
        os.environ['is_weibo'] = 'True'

    config_data = importlib.import_module(config_dir + 'data_config')
    config_model = importlib.import_module(config_dir + agent_name)
    model = importlib.import_module('model.' + agent_name)
    predictor = model.Predictor(config_model, config_data, 'test')

    init_start_time = time.time()
    print("生成 TGODC-{}-{} Model 实例.................".format(agent_name, dataset))
    target_chat_instance = Target_Chat(model, config_model, config_data)
    print("TGODC-{}-{} Model 实例生成完成...............".format(agent_name, dataset))
    init_end_time = time.time()
    print('初始化花费时间: {:.2f}s'.format(init_end_time - init_start_time))

    return target_chat_instance
