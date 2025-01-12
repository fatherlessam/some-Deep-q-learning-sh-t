import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import random
import matplotlib.pyplot as plt
from collections import deque


class HomeEnvironment:
    def __init__(self):
        self.nodes = ["home", "v1", "v2", "d1", "d2"]
        self.edges = {
            "home": {"forward": "v1"},
            "v1": {"left": "v2", "right": "d1"},
            "v2": {"forward": "d2"}
        }
        self.positions = {
            "home": (0, 0),
            "v1": (1, 0),
            "v2": (1, -1),
            "d1": (1, 1),
            "d2": (2, -1)
        }
        self.goal = "d2"
        self.current_state = "home"
        self.visited_nodes = set()

    def reset(self):
        self.current_state = "home"
        self.visited_nodes = {"home"}
        return self.state_to_vector(self.current_state)

    def step(self, action):
        action_mapping = ["left", "right", "forward", "backward"]
        action_name = action_mapping[action]

        if action_name in self.edges.get(self.current_state, {}):
            next_state = self.edges[self.current_state][action_name]
        else:
            next_state = "unknown"

        reward = self.reward_function(next_state)
        done = next_state == self.goal
        if next_state != "unknown":
            self.visited_nodes.add(next_state)
        self.current_state = next_state if next_state != "unknown" else self.current_state
        return self.state_to_vector(self.current_state), reward, done

    def reward_function(self, next_state):
        if next_state == self.goal:
            return 100
        elif next_state == "unknown":
            return -100
        elif next_state in self.visited_nodes:
            return -10
        else:
            return -1

    def state_to_vector(self, state):
        vector = np.zeros(len(self.nodes))
        if state in self.nodes:
            vector[self.nodes.index(state)] = 1
        return vector

    def get_position(self, state):
        return self.positions.get(state, None)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = HomeEnvironment()
    state_size = len(env.nodes)
    action_size = 4  
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    scores = [] 

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                print(f"episode: {e}/{episodes}, score: {time}, reward: {total_reward}, e: {agent.epsilon:.2}")
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        scores.append(total_reward)


    plt.plot(scores)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    print("Training completed.")
