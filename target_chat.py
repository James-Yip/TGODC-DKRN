import tensorflow as tf
import importlib
import random
from preprocess.data_utils import utter_preprocess, is_reach_goal
from utils.log_utils import create_logs, add_log

class Target_Chat:
    def __init__(self,
                 agent,
                 target_set,
                 start_corpus,
                 max_turns,
                 conversation_save_path):
        self.agent = agent
        self.target_set = target_set
        self.start_corpus = start_corpus
        self.max_turns = max_turns
        self.conversation_save_path = conversation_save_path
        self.current_sessions = 0

        self.sess = tf.Session(config=self.agent.gpu_config)
        self.agent.retrieve_init(self.sess)
        create_logs(self.conversation_save_path)

    def chat(self, user_input=None):
        responses = []
        # if is the beginning of a conversation
        if user_input is None:
            self._reset()
            reply = self.start_utterance
            add_log(self.conversation_save_path, '-------- Session {} --------'.format(self.current_sessions))
            add_log(self.conversation_save_path, 'START: {}'.format(reply))
        else:
            self.history.append(user_input)
            source = utter_preprocess(self.history, self.agent.data_config._max_seq_len)
            reply = self.agent.retrieve(source, self.sess)
            add_log(self.conversation_save_path, 'HUMAN: {}'.format(user_input), print_details=False)
            add_log(self.conversation_save_path, 'AGENT: {}'.format(reply))
        self.history.append(reply)
        responses.append(reply)

        # if the last two utterances contain target keyword
        if is_reach_goal(' '.join(self.history[-2:]), self.target_keyword):
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
        self.history = []
        self.start_utterance = random.sample(self.start_corpus, 1)[0]
        self.target_keyword = random.sample(self.target_set,1)[0]
        self.agent.target = self.target_keyword
        self.agent.score = 0.
        self.agent.reply_list = []

def init_target_chat(agent_name):
    config_data = importlib.import_module('config.data_config')
    config_model = importlib.import_module('config.' + agent_name)
    model = importlib.import_module('model.' + agent_name)
    predictor = model.Predictor(config_model, config_data, 'test')

    print("生成 TGODC-{} Model 实例.................".format(agent_name))
    target_chat_instance = Target_Chat(predictor,
                                    config_data._test_keywords_candi,
                                    config_data._start_corpus,
                                    config_data._max_turns,
                                    config_model._conversation_save_path)
    print("TGODC-{} Model 实例生成完成...............".format(agent_name))
    return target_chat_instance

if __name__ == '__main__':
    flags = tf.flags
    flags.DEFINE_string('agent', 'neural_dkr', 'The agent type, supports neural_dkr / kernel / matrix / neural / retrieval / retrieval_stgy.')
    flags.DEFINE_integer('times', 10, 'Conversation times.')
    FLAGS = flags.FLAGS
    target_chat_instance = init_target_chat(FLAGS.agent)
    for i in range(FLAGS.times):
        responses = []
        target_chat_instance.chat()
        while len(responses) < 2:
            responses = target_chat_instance.chat(input('HUMAN: '))
