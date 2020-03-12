import tensorflow as tf
import importlib
import random
import os
from tqdm import tqdm
from preprocess.data_utils import utter_preprocess, is_reach_goal
from model import retrieval
from utils.log_utils import create_logs, add_logs, add_log

class Target_Simulation():
    def __init__(self, model, config_model, config_retrieval, config_data):
        g1 = tf.Graph()
        with g1.as_default():
            self.agent = model.Predictor(config_model, config_data, 'test')
            self.agent_sess = tf.Session(graph=g1, config=self.agent.gpu_config)
            self.agent.retrieve_init(self.agent_sess)
        g2 = tf.Graph()
        with g2.as_default():
            self.simulator = retrieval.Predictor(config_retrieval, config_data)
            self.simulator_sess = tf.Session(graph=g2, config=self.simulator.gpu_config)
            self.simulator.retrieve_init(self.simulator_sess)

        self.target_set = config_data._target_keywords_for_simulation
        self.start_corpus = config_data._start_corpus
        self.random_selected_sub_start_corpus = config_data._start_corpus_for_simulation
        self.max_turns = config_data._max_turns
        self.simulation_save_path = config_model._simulation_save_path
        create_logs(self.simulation_save_path)

    def self_play_simulation(self, simulation_cnt, start_utterance=None, print_details=True):
        if start_utterance is None:
            start_utterance = random.sample(self.start_corpus, 1)[0]
        simulation_start_str = 'start self-play simulation with start utterance: {} (total {} sessions)'.format(
            start_utterance, simulation_cnt)
        add_log(self.simulation_save_path, simulation_start_str, print_details)
        success_cnt, turns_cnt = 0, 0
        for i in tqdm(range(simulation_cnt)):
            add_log(self.simulation_save_path, '-------- Session {} --------'.format(i), print_details)
            success, turns = self.simulate(start_utterance=start_utterance,
                                           target_keyword=self.target_set[i],
                                           print_details=print_details)
            success_cnt += success
            turns_cnt += turns
        # the average number of turns used to reach a target
        success_rate = (success_cnt / simulation_cnt) * 100
        average_turns = turns_cnt / success_cnt
        simulation_result_str = '#success / #sessions: {}/{}, success rate: {:.1f}%, average turns: {:.2f}'.format(
                    success_cnt, simulation_cnt, success_rate, average_turns)
        add_log(self.simulation_save_path, simulation_result_str, print_details=True)
        return success_cnt, turns_cnt, simulation_result_str

    def self_play_simulation_with_fixed_start_corpus(self, simulation_cnt, print_details=True):
        success_cnt_list, turns_cnt_list, simulation_result_list = [], [], []
        for start_utterance in self.random_selected_sub_start_corpus:
            success_cnt, turns_cnt, simulation_result_str = \
                self.self_play_simulation(simulation_cnt, start_utterance, print_details)
            success_cnt_list.append(success_cnt)
            turns_cnt_list.append(turns_cnt)
            simulation_result_list.append(simulation_result_str)

        total_simulation_cnt = simulation_cnt * len(self.random_selected_sub_start_corpus)
        total_success_cnt = sum(success_cnt_list)
        total_turns_cnt = sum(turns_cnt_list)
        total_success_rate = (total_success_cnt / total_simulation_cnt) * 100
        total_average_turns = total_turns_cnt / total_success_cnt

        for start_utterance, simulation_result in \
            zip(self.random_selected_sub_start_corpus, simulation_result_list):
            add_log(self.simulation_save_path,
                    'For start utterance: {}, {}'.format(start_utterance, simulation_result),
                    print_details=True)
        add_log(self.simulation_save_path,
                'total success times / total sessions: {}/{}, total success rate: {:.1f}%, total average turns: {:.2f}'.format(
                    total_success_cnt, total_simulation_cnt, total_success_rate, total_average_turns),
                print_details=True)

    def simulate(self, start_utterance, target_keyword, print_details):
        history = []
        simulation_outputs = []
        history.append(start_utterance)
        self.agent.target = target_keyword
        self.agent.score = 0.
        self.agent.reply_list = []
        self.simulator.reply_list = []

        simulation_outputs.append('START: {}'.format(start_utterance))
        for i in range(self.max_turns):
            source = utter_preprocess(history, self.agent.data_config._max_seq_len)
            simulator_reply = self.simulator.retrieve(source, self.simulator_sess)
            history.append(simulator_reply)
            source = utter_preprocess(history, self.agent.data_config._max_seq_len)
            agent_reply = self.agent.retrieve(source, self.agent_sess)
            simulation_outputs.append('SIMULATOR: {}'.format(simulator_reply))
            simulation_outputs.append('AGENT: {}'.format(agent_reply))
            if hasattr(self.agent, 'next_kw'):
                simulation_outputs.append('Keyword: {}, Similarity: {:.2f}'.format(self.agent.next_kw, self.agent.score))
            history.append(agent_reply)
            if is_reach_goal(' '.join(history[-2:]), target_keyword):
                simulation_outputs.append('[SUCCESS] target: \'{}\'.'.format(target_keyword))
                add_logs(self.simulation_save_path, simulation_outputs, print_details)
                return True, (len(history) + 1) // 2

        simulation_outputs.append('[FAIL] out of the max dialogue turns, target: \'{}\'.'.format(target_keyword))
        add_logs(self.simulation_save_path, simulation_outputs, print_details)
        return False, 0

if __name__ == '__main__':
    flags = tf.flags
    flags.DEFINE_string('dataset', 'TGPC', 'The dataset, supports TGPC / CWC.')
    flags.DEFINE_string('agent', 'neural_dkr', 'The agent type, \
        supports neural_dkr / kernel / matrix / neural / retrieval / retrieval_stgy.')
    flags.DEFINE_integer('times', 500, 'Simulation times.')
    flags.DEFINE_boolean('use_fixed_start_corpus', True, 'Whether to use fixed start_utterances.')
    flags.DEFINE_boolean('print_details', True, 'Whether to print simulation details or not.')
    FLAGS = flags.FLAGS

    # Target-Guided PersonaChat Dataset
    if FLAGS.dataset == 'TGPC':
        config_dir = 'config.'
        os.environ['is_weibo'] = 'False'
    # Chinese Weibo Conversation Dataset
    elif FLAGS.dataset == 'CWC':
        config_dir = 'config_weibo.'
        os.environ['is_weibo'] = 'True'

    config_data = importlib.import_module(config_dir + 'data_config')
    config_model = importlib.import_module(config_dir + FLAGS.agent)
    config_retrieval = importlib.import_module(config_dir + 'retrieval')
    model = importlib.import_module('model.' + FLAGS.agent)

    target_simulation_instance = Target_Simulation(model, config_model, config_retrieval, config_data)
    if FLAGS.use_fixed_start_corpus:
        target_simulation_instance.self_play_simulation_with_fixed_start_corpus(FLAGS.times, FLAGS.print_details)
    else:
        target_simulation_instance.self_play_simulation(FLAGS.times, print_details=FLAGS.print_details)
