import laserhockey.hockey_env as h_env
import numpy as np
import os
import wandb
from utils.evaluate_func import evaluate


class Trainer():
    def __init__(self, agent, save_path, save_freq, continue_training_step, path_pretrained, eval_freq, eval_eps, train_freq, train_n_steps, run_args):
        self.agent = agent
        self.save_path = save_path
        self.save_freq = save_freq
        self.continue_training_step = continue_training_step
        self.path_pretrained = path_pretrained
        self.eval_freq = eval_freq
        self.eval_eps = eval_eps
        self.train_freq = train_freq
        self.train_n_steps = train_n_steps
        self.run_args = run_args
        self.load_pretrained = False

        # pretrained model
        if self.continue_training_step > 0:
            self.agent.load_agent(self.path_pretrained, self.continue_training_step)
            print(f'continue training at step {self.continue_training_step}')
            self.pretrained = True
        
    def train(self, steps):
        if self.load_pretrained:
            print(f'Loading pretrained model from {self.path_pretrained} ...')

         # logger
        os.environ['WANDB_API_KEY'] = '3ef8f442aaa9ae9501ce542fcc6fd9e583ce78f8'

        self.save_path = os.path.abspath(self.save_path) + '/'
        os.makedirs(self.save_path, exist_ok=True)

        wandb.init(project='laser-hockey', config=self.run_args, dir=self.save_path)

        # get the run id by wandb
        run_id = wandb.run.id
        self.save_path += run_id

        # environment
        env = h_env.HockeyEnv(verbose=False)  # mode=h_env.HockeyEnv.TRAIN_DEFENSE, h_env.HockeyEnv.TRAIN_SHOOTING

        # training loop
        observation, _ = env.reset()
        observation_agent_2 = env.obs_agent_two()
        opponent = h_env.BasicOpponent(weak=True)  # weak opponent
        episode_reward = 0.0
        episode_reward_history = []
        for step in range(1, steps + self.continue_training_step + 1):
            step += self.continue_training_step

            action = self.agent.act(observation)
            action_agent_2 = opponent.act(observation_agent_2)
            new_observation, reward, done, _, info = env.step(np.hstack((action, action_agent_2)))

            # augmented reward
            '''
            reward_closeness_to_puck = info['reward_closeness_to_puck']
            reward_touch_puck = info['reward_touch_puck']
            reward_puck_direction = info['reward_puck_direction']
            reward += (reward_closeness_to_puck + reward_touch_puck + reward_puck_direction)
            '''
            

            episode_reward += reward
            self.agent.store_transition(observation, action, reward, new_observation, done)

            # Don't train at every step, but then train multiple times
            '''
            if step % self.train_freq == 0:
                for _ in range(self.train_n_steps):
                    lr = self.agent.train()
                    wandb.log({'lr': lr})
            '''

            observation = new_observation
            observation_agent_2 = env.obs_agent_two()

            self.agent.train()

            if done:
                observation = env.reset()[0]
                episode_reward_history.append(episode_reward)
                episode_reward = 0.0

            if step % self.save_freq == 0:
                self.agent.save_agent(self.save_path + '/models/', step)

            if step % 1000 == 0:
                print(f'Step: {step}, Reward: {np.round(episode_reward, 2)}')

            if step % self.eval_freq == 0:
                won_games, lost_games, drawn_games, avg_reward, avg_ep_length, avg_touched_pucks = evaluate(self.agent, env, opponent, self.eval_eps, step, save_path="plots/" + run_id, save_gifs=True, verbose=True)
                winrate, lossrate, drawrate = [n / self.eval_eps for n in [won_games, lost_games, drawn_games]]
                wandb.log({'winrate': winrate, 'lossrate': lossrate, 'drawrate': drawrate, 'avg_reward per ep': avg_reward, 'avg episode length': avg_ep_length, 'step': step, "avg_touched_pucks | puck starts in our half": avg_touched_pucks})

        # save the final agent
        self.agent.save_agent(self.save_path + '/models/', step)

        # save the reward history
        np.save(self.save_path + '/reward_history.npy', episode_reward_history)

        # save the run config
        with open(self.save_path + '/run_config.txt', 'w') as f:
            print(self.run_args, file=f)
