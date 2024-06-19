import gymnasium as gym
from gymnasium import spaces
import argparse
import os
import wandb
import imageio
import numpy as np

import utils
from dqn_agent import DQNAgent


def main(args):
    # logger
    os.environ['WANDB_API_KEY'] = '3ef8f442aaa9ae9501ce542fcc6fd9e583ce78f8'
    args.save_path = os.path.abspath(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    wandb.init(project=f'{args.env_name}-dqn', config=args, dir=args.save_path)

    # Environment
    switch={"pendulum":gym.make("Pendulum-v1", render_mode="rgb_array"),
            "cartpole":gym.make('CartPole-v1', render_mode="rgb_array"),
            }
    env = switch.get(args.env_name, "Environment not found") 

    if isinstance(env.action_space, spaces.Box):
        env = utils.DiscreteActionWrapper(env,5)

    ac_space = env.action_space
    o_space = env.observation_space
    print(ac_space)
    print(o_space)
    print(list(zip(env.observation_space.low, env.observation_space.high)))

    q_agent = DQNAgent(o_space, ac_space, discount=args.gamma, eps=args.eps_max, eps_min=args.eps_min, eps_decay=args.eps_decay, 
                    use_target_net=args.use_target, update_target_every=args.target_update_freq, dueling=args.use_dueling, clipped=args.use_clipping)

    ob,_info = env.reset()
    stats = []
    losses = []
    max_episodes=args.train_episodes
    max_steps=args.max_steps_per_episode 
    save_gifs=True
    frames = []
    total_steps = 0
    for i in range(max_episodes):
        total_reward = 0
        ob, _ = env.reset()
        for t in range(max_steps):
            total_steps += 1

            # Save a gif of every 500th episode
            if i%500==0 and save_gifs:
                frame = env.render()
                frames.append(utils._label_with_episode_number(frame, episode_num=i))

            a = q_agent.act(ob)
            (ob_new, reward, done, _, _) = env.step(a)
            total_reward+= reward
            q_agent.store_transition((ob, a, reward, ob_new, done))            
            ob=ob_new        
            q_agent.update_eps(total_steps)
            if done: 
                break
        losses.extend(q_agent.train(32))
        stats.append([i,total_reward,t+1])    
        wandb.log({"Reward": total_reward, "Episode": i, "Episode length": stats[-1][2]})

        if i%100==0 and i>0:
            print(f"Episode: {i} / {max_episodes}")

        if i%500==0 and save_gifs:
            gif_path = os.path.join(args.save_path, f'./gifs/train_episode{i}/')
            os.makedirs(gif_path, exist_ok=True)
            imageio.mimwrite(os.path.join(gif_path, f'episode_{i}.gif'), frames, duration=90)
        running_mean_reward = np.mean(np.array(stats[-50:]), axis=0)[1] if i>50 else 0

    # After training, save model
    q_agent.save_agent(path=args.save_path, step=total_steps)
    print(total_steps)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train an Agent')
    parser.add_argument('--train_episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--max_steps_per_episode', type=int, default=500, help='Maximum number of steps per episode')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--eps_max', type=float, default=1.0, help='Start eps for epsilon-greedy exploration')
    parser.add_argument('--eps_min', type=float, default=0.01, help='End eps for epsilon-greedy exploration')
    parser.add_argument('--eps_decay', type=float, default=0.99, help='Decay period from start to end epsilon')
    parser.add_argument('--target_update_freq', type=int, default=20, help='After how many train steps to update target network') 
    parser.add_argument('--save_path', type=str, default='saved_models/dqn/', help='Path to save models etc.')
    parser.add_argument('--path_pretrained', type=str, default='saved_models/dqn/', help='Path to pretrained model')
    parser.add_argument('--continue_training_step', type=int, default=0, help='Continue training at step x')
    parser.add_argument('--env_name', type=str, default='pendulum', help='Environment to train in')
    parser.add_argument('--convergence_threshold', type=int, default=200, help='Avg reward needed to solve environment')
    parser.add_argument('--save_gifs', type=bool, default=False, help='Save gifs')
    parser.add_argument('--use_target', type=bool, default=True, help='Use target network (Double DQN)')
    parser.add_argument('--use_dueling', type=bool, default=True, help='Use dueling architecture')
    parser.add_argument('--use_clipping', type=bool, default=True, help='Use clipping for TD targets')
    
    args = parser.parse_args()
    main(args)
