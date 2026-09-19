import flappy_bird_gymnasium
import gymnasium as gym
from experience_replay import ReplayMemory 
from dqn import DQN
import torch
import itertools
import yaml 
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random
import re

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
    
RUNS_DIR='runs'
os.makedirs(RUNS_DIR,exist_ok=True)

class Agent:
    def __init__(self,param_set):
        self.param_set=param_set
        with open("parameters.yaml",'r')as f:
            all_param_set=yaml.safe_load(f)
            params=all_param_set[param_set]

        self.alpha=params["alpha"]
        self.gamma=params["gamma"]
        self.epsilon_init=params["epsilon_init"]
        self.epsilon_min= params["epsilon_min"]
        self.epsilon_decay=params["epsilon_decay"]
        self.replay_memory_size=params["replay_memory_size"]
        self.mini_batch_size=params["mini_batch_size"]
        self.network_sync_rate=params["network_sync_rate"]
        self.reward_threshold=params["reward_threshold"]
        self.loss_fn=nn.MSELoss()
        self.optimizer=None
        self.LOG_FILE=os.path.join(RUNS_DIR,f"{self.param_set}.log")
        self.MODEL_FILE=os.path.join(RUNS_DIR,f'{self.param_set}.pt')

    def run(self,is_training=True,render=False):
        env=gym.make("FlappyBird-v0",render_mode="human" if render else None)   
        num_states=env.observation_space.shape[0]
        num_actions=env.action_space.n
        policy_dqn=DQN(num_states,num_actions).to(device)
        state,_=env.reset()
        
        start_episode = 0
        best_reward = float("-inf")
        
        if is_training:
            epsilon = self.epsilon_init
            if os.path.exists(self.MODEL_FILE):
                checkpoint = torch.load(self.MODEL_FILE)
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    # New-style checkpoint: also restore epsilon so we don't
                    # go back to near-random exploration every time we resume.
                    policy_dqn.load_state_dict(checkpoint["model_state"])
                    epsilon = checkpoint.get("epsilon", self.epsilon_init)
                else:
                    # Old-style checkpoint (just a raw state_dict). Fall back
                    # to epsilon_init since we have no saved epsilon to restore.
                    policy_dqn.load_state_dict(checkpoint)
                print(f"Resumed from checkpoint: {self.MODEL_FILE} (epsilon={epsilon:.4f})")
            
            memory=ReplayMemory(self.replay_memory_size)
            target_dqn=DQN(num_states,num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps=0
            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)
            
            if os.path.exists(self.LOG_FILE):
                with open(self.LOG_FILE,"r") as f:
                    lines=f.readlines()
                    # Walk backwards from the end of the file to find the last
                    # line that actually matches the expected pattern, in case
                    # the file ends with a blank line or a partially-written
                    # line from an interrupted run.
                    for last_line in reversed(lines):
                        match = re.search(r"best reward=([0-9\.-]+) for episode=([0-9]+)", last_line)
                        if match:
                            best_reward = float(match.group(1))
                            start_episode = int(match.group(2))
                            break
        else:
            epsilon = 0 
            checkpoint = torch.load(self.MODEL_FILE)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                policy_dqn.load_state_dict(checkpoint["model_state"])
            else:
                policy_dqn.load_state_dict(checkpoint)
            policy_dqn.eval()
            
        for episode in itertools.count(start=start_episode):
            state,_=env.reset()
            state=torch.tensor(state,dtype=torch.float,device=device)
            episode_reward=0
            terminated=False
            
            while not terminated and episode_reward<self.reward_threshold :
                if is_training and random.random()<epsilon:
                    action=env.action_space.sample()
                    action=torch.tensor(action, dtype=torch.long, device=device) 
                else:
                    with torch.no_grad():
                        action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                                  
                next_state,reward,terminated,_,_=env.step(action.item())
                episode_reward+=reward
                reward=torch.tensor(reward,dtype=torch.float,device=device)
                next_state=torch.tensor(next_state,dtype=torch.float,device=device)
                
                if is_training :
                    terminated_t = torch.tensor(terminated, dtype=torch.float, device=device)
                    memory.append((state,action,next_state,reward,terminated_t))
                    steps+=1
                    
                    # Epsilon decay happens per-step
                    epsilon=max(self.epsilon_decay * epsilon, self.epsilon_min)
                    
                state=next_state 
                
                if is_training and len(memory)>self.mini_batch_size:
                    mini_batch=memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch,policy_dqn,target_dqn)

                    if steps>=self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps=0
                        
            print(f"for episode={episode+1} total rewards={episode_reward} and epsilon={epsilon}")
            
            if is_training:
                if episode_reward>best_reward:
                    log_msg=f"best reward={episode_reward} for episode={episode+1}"
                    with open(self.LOG_FILE,"a") as f:
                        f.write(log_msg+"\n")
                    torch.save({
                        "model_state": policy_dqn.state_dict(),
                        "epsilon": epsilon,
                    }, self.MODEL_FILE)
                    best_reward=episode_reward

    def optimize(self,mini_batch,policy_dqn,target_dqn):
            states,actions,next_states,rewards,terminations =zip(* mini_batch)
            states=torch.stack(states)
            actions=torch.stack(actions)
            next_states=torch.stack(next_states)
            rewards=torch.stack(rewards)
            terminations = torch.tensor(terminations, dtype=torch.float, device=device)
            with torch.no_grad():
                target_q=rewards+(1-terminations)*self.gamma*target_dqn(next_states).max(dim=1)[0]
            current_q=policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()
            
            loss=self.loss_fn(current_q,target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train or test model")
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train',help='Training mode',action='store_true')
    args=parser.parse_args()

    dql=Agent(param_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False,render=True)