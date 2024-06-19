import laserhockey.hockey_env as h_env
import argparse
import os
import wandb
import numpy as np

from dqn_agent import DQNAgent
from evaluate_func import evaluate


def main(args):
    # logger
    os.environ['WANDB_API_KEY'] = '3ef8f442aaa9ae9501ce542fcc6fd9e583ce78f8'
    args.save_path = os.path.abspath(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    wandb.init(project=f'{args.env_name}-dqn', config=args, dir=args.save_path)

    # environment
    if args.mode == None:
        env = h_env.HockeyEnv(mode=0, verbose=False)
    elif args.mode == "train_shooting":
        env = h_env.HockeyEnv(mode=1, verbose=False)
    elif args.mode == "train_defense":
        env = h_env.HockeyEnv(mode=2, verbose=False)
    opponent = h_env.BasicOpponent(weak=True)

    ac_space = env.discrete_action_space
    o_space = env.observation_space
    print(ac_space)
    print(o_space)

    identifier = wandb.run.name
    target_update = 20
    q_agent = DQNAgent(o_space, ac_space, discount=args.gamma, eps=args.eps_start, eps_end=args.eps_end, eps_decay=args.eps_decay, 
                    use_target_net=args.use_target, update_target_every= target_update, dueling=args.use_dueling, cliping=args.use_clipping, identifier=identifier)

    if args.continue_training_step != 0:
        q_agent.load_agent(path=args.path_pretrained, step=args.continue_training_step)

    stats = []
    losses = []
    max_episodes=10000
    max_steps=500 
    total_steps = args.continue_training_step
    for i in range(max_episodes):
        total_steps += 1       
        ob, _ = env.reset()
        observation_agent_2 = env.obs_agent_two()
        total_reward = 0
        for t in range(max_steps): 
            a = q_agent.act(ob)
            cont_a = env.discrete_to_continous_action(a)
            action_agent_2 = opponent.act(observation_agent_2)
            (ob_new, reward, done, _, _) = env.step(np.hstack((cont_a, action_agent_2)))
            total_reward+= reward
            q_agent.store_transition((ob, a, reward, ob_new, done))            
            ob=ob_new  
            q_agent.update_eps(total_steps)      
            if done: break    
        losses.extend(q_agent.train(32))
        stats.append([i,total_reward,t+1])    
        
        if i%100==0 and i>0:
            print(f"Episode: {i} / {max_episodes}")

        if i%args.eval_freq==0:
            won_games, lost_games, drawn_games, avg_reward, avg_ep_length, avg_touched_pucks = evaluate(q_agent, env, opponent, args.eval_eps, i, args.save_path, save_gifs=True, verbose=True)
            wandb.log({'winrate': won_games, 'lossrate': lost_games, 'drawrate': drawn_games, 'avg_reward per ep': avg_reward, 'avg episode length': avg_ep_length, 'Episode': i, "avg_touched_pucks | puck starts in our half": avg_touched_pucks})

    # After training, save model
    q_agent.save_agent(path=args.save_path, step=total_steps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a Agent')
    parser.add_argument('--train_episodes', type=int, default=12_000, help='Number of episodes to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')#original batch_size=256
    parser.add_argument('--eps_start', type=float, default=1.0, help='Start eps for epsilon-greedy exploration')
    parser.add_argument('--eps_end', type=float, default=0.02, help='End eps for epsilon-greedy exploration')
    parser.add_argument('--eps_decay', type=float, default=0.995, help='Decay period from start to end epsilon')
    parser.add_argument('--target_update_freq', type=int, default=200, help='After how many train steps to update target network') 
    parser.add_argument('--max_buffer_size', type=int, default=500_000, help='Maximum transitions saved in replay buffer')
    parser.add_argument('--save_path', type=str, default='saved_models/dqn/', help='Path to save models etc.')
    parser.add_argument('--save_freq', type=int, default=100_000, help='Save model every x steps')
    parser.add_argument('--path_pretrained', type=str, default='saved_models/dqn/', help='Path to pretrained model')
    parser.add_argument('--continue_training_step', type=int, default=0, help='Continue training at step x')
    parser.add_argument('--eval_freq', type=int, default=1000, help='Evaluate every x episodes')
    parser.add_argument('--eval_eps', type=int, default=500, help='Number of episodes to evaluate')
    parser.add_argument('--mode', type=str, default=None, help='Training mode')
    parser.add_argument('--save_gifs', type=bool, default=True, help='Save gifs')
    parser.add_argument('--use_target', type=bool, default=True, help='Use target network (Double DQN)')
    parser.add_argument('--use_dueling', type=bool, default=True, help='Use dueling architecture')
    parser.add_argument('--use_clipping', type=bool, default=True, help='Use clipping for TD targets')

    args = parser.parse_args()
    main(args)