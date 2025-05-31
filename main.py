import gymnasium
from agent import QAgent
import torch
import os
import ale_py

gymnasium.register_envs(ale_py)

# agent1 = QAgent.QAgent(state_dim=8, action_num=4, hidden_dim=144, learning_rate=2e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent1 = QAgent.VideoQAgent(video_dim=(210, 160, 3), action_num=7, device=device)

# environment_name = "LunarLander-v2"
environment_name = "ALE/Assault-v5"

# env = gymnasium.make("LunarLander-v3", render_mode="human")
env1 = gymnasium.make(environment_name, render_mode="human", continuous=False)
env2 = gymnasium.make(environment_name, render_mode="human", continuous=False)


def train_agent(steps, env, agent, load_model_path=None, save_model_path=None):
    observation, info = env.reset(seed=42)
    accumulated_reward = 0
    steps_taken = 0
    # load network
    if load_model_path is not None:
        if os.path.exists(load_model_path):
            agent.load_network(load_model_path)
        else:
            print(f"Saved model {load_model_path} do not exist. Creating a new model ...")
    for _ in range(steps):
        action = agent.take_action(observation, epsilon=0.2)  # this is where you would insert your policy
        observation2, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"reward for the episode:{accumulated_reward}, taking {steps_taken} steps")
            accumulated_reward = 0
            steps_taken = 0
        else:
            accumulated_reward += reward
            steps_taken += 1
        agent.update(observation, observation2, action, reward, terminated)
        # Save network
        if (save_model_path is not None) and steps % 1000 == 0:
            agent.save_network(save_model_path)
        observation = observation2

        if terminated or truncated:
            observation, info = env.reset()


def validate_agent(steps, env, agent):
    observation, info = env.reset(seed=42)
    accumulated_reward = 0
    for _ in range(steps):
        action = agent.take_action(observation, epsilon=0)  # this is where you would insert your policy
        observation2, reward, terminated, truncated, info = env.step(action)
        accumulated_reward += reward
        observation = observation2
        env.render()

        if terminated or truncated:
            observation, info = env.reset()
            print(f"Episode ended. Reward:{accumulated_reward}")
            accumulated_reward = 0


if __name__ == "__main__":
    train_agent(400000, env1, agent1)
    # train_agent(200000, env1, agent1, "./Saved_model/LunarLander_model_1.pth", "./Saved_model/LunarLander_model_1.pth")
    validate_agent(2000, env2, agent1)
    env1.close()
