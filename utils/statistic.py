import os
import numpy as np
import tensorflow as tf

class Statistic(object):
    def __init__(self, sess, t_test, t_learn_start, model_path, dir_name):
        self.sess = sess
        self.t_test = t_test    # The maximum number of t while training (*= scale)
        self.t_learn_start = t_learn_start   # The time when to begin training (*= scale)

        self.reset()
        self.max_avg_episode_reward = 0

        with tf.variable_scope('t'):
            self.t_op = tf.Variable(0, trainable=False, name='t')
            self.t_add_op = self.t_op.assign_add(1)

        self.model_path = model_path
        self.writer = tf.summary.FileWriter('./tensorboard_logs/%s' % dir_name, self.sess.graph)

        with tf.variable_scope('summary'):
            scalar_summary_tags = [
                'average/reward',
                'episode/max_reward', 'episode/min_reward', 'episode/avg_reward',
                'episode/num_of_episode', 'episode/num_of_success', 'training/epsilon',
            ]

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.scalar(tag, self.summary_placeholders[tag])

            histogram_summary_tags = ['episode/rewards', 'episode/actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

    def reset(self):
        self.num_episode = 0
        self.num_success = 0
        self.update_count = 0
        self.episode_reward = 0.
        self.total_reward = 0.
        self.episode_rewards = []
        self.actions = []

    def set_saver(self, variables, max_to_keep=1):
        self.saver = tf.train.Saver(list(variables), max_to_keep=max_to_keep)

    def on_step(self, t, action, reward, terminal, success, epsilon, is_update):
        if t >= self.t_learn_start:
            self.actions.append(action)
            self.total_reward += reward

            if terminal:
                self.num_episode += 1
                if success:
                    self.num_success += 1
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0.
            else:
                self.episode_reward += reward

            if is_update:
                self.update_count += 1

            if (t + 1) % self.t_test == 0 and self.update_count != 0:
                avg_reward = self.total_reward / self.t_test

                try:
                    max_episode_reward = np.max(self.episode_rewards)
                    min_episode_reward = np.min(self.episode_rewards)
                    avg_episode_reward = np.mean(self.episode_rewards)
                except:
                    max_episode_reward, min_episode_reward, avg_episode_reward = 0, 0, 0

                print('\navg_r: %.4f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, #episode: %d, #success: %d' \
                       % (avg_reward, avg_episode_reward, max_episode_reward, min_episode_reward, self.num_episode, self.num_success))

                if self.max_avg_episode_reward * 0.9 <= avg_episode_reward:
                    assert t == self.get_t()

                    self.save_model(t)

                    self.max_avg_episode_reward = max(self.max_avg_episode_reward, avg_episode_reward)

                    self.inject_summary({
                        'average/reward': avg_reward,
                        'episode/max_reward': max_episode_reward,
                        'episode/min_reward': min_episode_reward,
                        'episode/avg_reward': avg_episode_reward,
                        'episode/num_of_episode': self.num_episode,
                        'episode/num_of_success': self.num_success,
                        'episode/actions': self.actions,
                        'episode/rewards': self.episode_rewards,
                        'training/epsilon': epsilon,
                    }, t)

                self.reset()
        self.t_add_op.eval(session=self.sess)

    def inject_summary(self, tag_dict, t):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, t)
        self.writer.flush()

    def get_t(self):
        return self.t_op.eval(session=self.sess)

    def save_model(self, t):
        print(" [*] Saving checkpoints into {}".format(self.model_path))
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.saver.save(self.sess, self.model_path, global_step=t)

    def load_model(self):
        model_dir = os.path.dirname(self.model_path)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(model_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.model_path)
            return False