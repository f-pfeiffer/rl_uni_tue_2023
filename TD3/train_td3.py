from agents.td3_agent import TD3Agent
import laserhockey.hockey_env as h_env
import torch
import argparse
from trainer import Trainer

def main(args):
    # environment
    env = h_env.HockeyEnv(verbose=False)
    input_dim = env.observation_space.shape[0]
    n_actions = int(env.action_space.shape[0] / 2)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    args.device = device
    print(f'Using device: {device}')

    # agent
    agent = TD3Agent(lr_critic=args.lr_critic,
                lr_actor=args.lr_actor,
                gamma=args.gamma,
                input_dim=input_dim,
                tau=args.tau,
                n_actions=n_actions,
                max_buffer_size=args.max_buffer_size,
                batch_size=args.batch_size,
                update_actor_interval=args.update_actor_interval,
                explore_n_times=args.explore_n_times,
                noise=args.noise,
                device=device)
    
    # trainer
    trainer = Trainer(agent=agent,
                        save_path=args.save_path,
                        save_freq=args.save_freq,
                        continue_training_step=args.continue_training_step,
                        path_pretrained=args.path_pretrained,
                        eval_freq=args.eval_freq,
                        eval_eps=args.eval_eps,
                        train_freq=args.train_freq,
                        train_n_steps=args.train_n_steps,
                        run_args=args)
    
    trainer.train(steps=args.steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Agent')
    parser.add_argument('--steps', type=int, default=4_000_000, help='Number of steps to train')
    parser.add_argument('--lr_critic', type=float, default=0.0003, help='Learning rate for actor')
    parser.add_argument('--lr_actor', type=float, default=0.0003, help='Learning rate for critic')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update factor')
    parser.add_argument('--max_buffer_size', type=int, default=1_000_000, help='Max size of replay buffer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--update_actor_interval', type=int, default=2, help='Update interval for actor network')
    parser.add_argument('--explore_n_times', type=int, default=1_000, help='Explore n steps with random actions before agent starts to predict')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise for the mu')
    parser.add_argument('--save_path', type=str, default='runs/td3/', help='Path to save models etc.')
    parser.add_argument('--save_freq', type=int, default=50_000, help='Save model every x steps')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    parser.add_argument('--continue_training_step', type=int, default=0, help='Continue training at step x')
    parser.add_argument('--path_pretrained', type=str, default='runs/td3/hsn6fbe0/models/', help='Path to pretrained model')
    parser.add_argument('--eval_freq', type=int, default=50_000, help='Evaluate every x steps')
    parser.add_argument('--eval_eps', type=int, default=500, help='Number of episodes to evaluate')
    parser.add_argument('--train_freq', type=int, default=1, help='Train every x steps')
    parser.add_argument('--train_n_steps', type=int, default=1, help='Train for x steps')
    parser.add_argument('--save_gifs', action='store_true', help='Save gifs of evaluation episodes')
    args = parser.parse_args()
    main(args)