import csv
import os

class TrainingLogger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)

        # Episode-level log
        self.episode_path = os.path.join(log_dir, 'episode_log.csv')
        self.episode_file = open(self.episode_path, 'w', newline='')
        self.episode_writer = csv.writer(self.episode_file)
        self.episode_writer.writerow([
            'episode', 'total_reward', 't1_fighter_reward',
            't2_fighter_reward', 'epsilon', 'steps', 'buffer_size'
        ])

        # Step-level loss log
        self.loss_path = os.path.join(log_dir, 'loss_log.csv')
        self.loss_file = open(self.loss_path, 'w', newline='')
        self.loss_writer = csv.writer(self.loss_file)
        self.loss_writer.writerow(['global_step', 'loss'])

    def log_episode(self, episode, total_reward, t1_reward, t2_reward,
                    epsilon, steps, buffer_size):
        self.episode_writer.writerow([
            episode, round(total_reward, 4), round(t1_reward, 4),
            round(t2_reward, 4), round(epsilon, 4), steps, buffer_size
        ])
        self.episode_file.flush()  # write immediately, don't wait for close

    def log_loss(self, global_step, loss):
        self.loss_writer.writerow([global_step, round(loss, 6)])
        self.loss_file.flush()

    def close(self):
        self.episode_file.close()
        self.loss_file.close()