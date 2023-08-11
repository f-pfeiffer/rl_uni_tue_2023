import numpy as np
import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt    


def evaluate(agent, env, opponent, eval_episodes, train_step, save_path, save_gifs=False, verbose=True):

    won_games = 0
    lost_games = 0
    drawn_games = 0
    ep_rewards = []
    ep_length = []
    ep_touched_puck = []

    for episode_counter in range(eval_episodes):
        episode_reward = 0
        episode_length = 0
        obs, _ = env.reset()
        obs_opp = env.obs_agent_two()

        agent_touched_puck = False
        puck_starts_in_our_half = True if env.puck.position[0] < 5 else False

        save_gifs = save_gifs and episode_counter < 5

        if save_gifs:
            frames = []

        for step in range(env.max_timesteps + 1):
                if env.player1_has_puck and puck_starts_in_our_half:
                    agent_touched_puck = True

                if hasattr(agent, 'is_dqn'): # a little extra sausage for dqn
                    eps = agent.get_current_eps(step)
                    agent_move = agent.act(obs, eps)
                    discr_agent_move = agent.act(obs, eps)
                    agent_move = env.discrete_to_continous_action(discr_agent_move)
                else:
                    agent_move = agent.act(obs)

                opp_move = opponent.act(obs_opp)

                if save_gifs:
                    frame = env.render(mode='rgb_array')
                    frames.append(_label_with_episode_number(frame, episode_num=episode_counter))

                (obs_next, reward, done, _, _) = env.step(np.hstack([agent_move, opp_move]))

                obs = obs_next
                obs_opp = env.obs_agent_two()
                episode_reward += reward
                episode_length += 1

                if done:
                    if save_gifs:
                        gif_path = os.path.join(save_path, f'./gifs/train_step{train_step}/')
                        os.makedirs(gif_path, exist_ok=True)
                        imageio.mimwrite(os.path.join(gif_path, f'episode_{episode_counter}.gif'), frames)
                        
                    #print("Eval Episode finished after {} timesteps".format(step+1))
                    ep_rewards.append(episode_reward)
                    ep_length.append(episode_length)
                    ep_touched_puck.append(agent_touched_puck)
                    episode_length = 0
                    if env.winner == 1:
                        won_games += 1
                    elif env.winner == -1:
                        lost_games += 1
                    else:
                        drawn_games += 1
                    break
                

    if verbose:
        print("Evaluation stats:")
        print(f'Won: {won_games}, Lost: {lost_games}')
        print(f'Winrate: {np.round(won_games / eval_episodes, 2)}')
        print(f'Average reward: {np.round(np.mean(ep_rewards), 2)}')
        print(f'Average episode length: {np.round(np.mean(ep_length), 2)}')
    
    return (
        won_games,
        lost_games,
        drawn_games,
        np.mean(ep_rewards),
        np.mean(ep_length),
        np.mean(ep_touched_puck)
    )



def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im


def save_random_agent_gif(env):
    frames = []
    for i in range(5):
        state = env.reset()        
        for t in range(500):
            action = env.action_space.sample()

            frame = env.render(mode='rgb_array')
            frames.append(_label_with_episode_number(frame, episode_num=i))

            state, _, done, _ = env.step(action)
            if done:
                break

    env.close()

    imageio.mimwrite(os.path.join('./videos/', 'random_agent.gif'), frames, fps=60)