import torch
import numpy as np
import gym

from rl_algo.normalization import Normalization, RewardScaling
from rl_algo.replaybuffer import ReplayBuffer
from rl_algo.ppo_continuous import PPO_continuous

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



def save_model_param(PPO_config, actor_net, critic_net, actor_opt, critic_opt, current_step, path):
    state_dict = {
        'PPO_config': PPO_config, 
        'actor_net': actor_net.state_dict(), 
        'critic_net': critic_net.state_dict(),
        'actor_opt': actor_opt.state_dict(),
        'critic_opt': critic_opt.state_dict(),
        'current_step': current_step
        }
    torch.save(state_dict, path)
    
    
def load_model_param(path):
    checkpoint = torch.load(path)
    return checkpoint

    # TODO next:
    # current_step = checkpoint['current_step']
    # network.load_state_dict(checkpoint['network'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    

def evaluate_policy(PPO_config, env, agent, state_norm):
    """
    Evaluate policy 3 times
    """
    times = 3
    # print('To evaluate {} times'.format(times))
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if PPO_config.use_state_norm:
            s = state_norm(s[0], update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if PPO_config.policy_dist == "Beta":
                action = 2 * (a - 0.5) * PPO_config.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, done, truncated, _ = env.step(action)
            done = done or truncated
            
            if PPO_config.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(PPO_config, quadruppedEnv):
    env = quadruppedEnv
    env_name = "Amphibious_Quadrupped"
    # env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    # env.seed(seed)
    seed = PPO_config.seed
    env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    PPO_config.state_dim = env.observation_space.shape[0]
    PPO_config.action_dim = env.action_space.shape[0]
    PPO_config.max_action = float(env.action_space.high[0])
    PPO_config.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(PPO_config.state_dim))
    print("action_dim={}".format(PPO_config.action_dim))
    print("max_action={}".format(PPO_config.max_action))
    print("max_episode_steps={}".format(PPO_config.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(PPO_config)
    agent = PPO_continuous(PPO_config)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_seed_{}'.format(env_name, PPO_config.policy_dist, seed))

    state_norm = Normalization(shape=PPO_config.state_dim)  # Trick 2:state normalization
    if PPO_config.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif PPO_config.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=PPO_config.gamma)

    last_eval_step = 0
    
    with tqdm(total=PPO_config.max_train_steps, desc='PPO Trainning', leave=True, ncols=80, unit='steps', unit_scale=True) as pbar:
        
        while total_steps < PPO_config.max_train_steps:
            s = env.reset()
            if PPO_config.use_state_norm:
                s = state_norm(s[0])
            if PPO_config.use_reward_scaling:
                reward_scaling.reset()
            episode_steps = 0
            done = False
            while not done:
            
                episode_steps += 1
                
                a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
                if PPO_config.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * PPO_config.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                # s_, r, done, _ = env.step(action)
                s_, r, done, truncated, _ = env.step(action)
                done = done or truncated
                
                # print('episode_steps:',episode_steps,'      done:',done)

                if PPO_config.use_state_norm:
                    s_ = state_norm(s_)
                if PPO_config.use_reward_norm:
                    r = reward_norm(r)
                elif PPO_config.use_reward_scaling:
                    r = reward_scaling(r)

                # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
                # dw means dead or win,there is no next state s';
                # but when reaching the max_episode_steps,there is a next state s' actually.
                if done and episode_steps != PPO_config.max_episode_steps:
                    dw = True
                else:
                    dw = False

                # Take the 'action'，but store the original 'a'（especially for Beta）
                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                s = s_
                
                total_steps += 1
                pbar.update(1)
                
                # print('total_steps:',total_steps)

                # When the number of transitions in buffer reaches batch_size,then update
                if replay_buffer.count == PPO_config.batch_size:
                    
                    # time.sleep(1)
                    tqdm.write("\n batch completed! steps: {}/{}".format(total_steps, PPO_config.max_train_steps))
                    # print("\n batch completed! steps: {}/{}".format(total_steps, PPO_config.max_train_steps))
                    
                    agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0



            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps - last_eval_step >= PPO_config.evaluate_freq:
                
                last_eval_step = total_steps
                # time.sleep(1)
                s = env.reset()
                if PPO_config.use_state_norm:
                    s = state_norm(s[0])
                if PPO_config.use_reward_scaling:
                    reward_scaling.reset()
                
                evaluate_num += 1
                evaluate_reward = evaluate_policy(PPO_config, env, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                
                tqdm.write("\n evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                # print("\n evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % PPO_config.save_freq == 0:
                    np.save('./data_train/PPO_continuous_{}_env_{}_seed_{}.npy'.format(PPO_config.policy_dist, env_name, seed), np.array(evaluate_rewards))
                
                if evaluate_num % PPO_config.save_freq == 0:
                    save_model_param(
                        PPO_config=PPO_config, 
                        actor_net=agent.actor,
                        critic_net=agent.critic,
                        actor_opt=agent.optimizer_actor,
                        critic_opt=agent.optimizer_critic,
                        current_step=total_steps,
                        path="./checkpoint/PPO_model_steps_{}_evaluate_{}".format(total_steps, evaluate_num)
                        )
