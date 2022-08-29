from Snake_game import Snake
from collections import deque
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import random
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam, RMSprop

def create_model(input_shape, action_space):
    """Create Deep Q learning model"""
    X_input = Input(input_shape)
    
    # Neural network layers and neurons
    X = Dense(units=128, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform')(X_input)
    X = Dense(units=128, activation='relu', kernel_initializer='he_uniform')(X)
    X = Dense(units=128, activation='relu', kernel_initializer='he_uniform')(X)
    
    #Output layer 4 units one for each direction
    X = Dense(units=action_space, activation='softmax', kernel_initializer='he_uniform')(X)
    
    model = Model(inputs=X_input, outputs=X)
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    
    return model

class DQNAgent:
    """
    Create DQNAgent that will train on snake data
    """
    
    def __init__(self):
        self.env = Snake()
        self.state_size = self.env.state_space
        self.action_space = self.env.action_space
        self.EPISODES = 1000
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters, should be turned into arguments and documented, but it works for now and is somewhat easy to change
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9995
        self.batch_size = 64
        self.train_start = 1000
        
        # Create model
        self.model = create_model(input_shape=(self.state_size,), action_space=self.action_space)
        
    # Append to the memory and decreases epsilon
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min: 
            self.epsilon *= self.epsilon_decay
            
    def act(self, state):
        # If random number is less than epsilon, then choose random action, this allows for exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        return np.argmax(self.model.predict(state))
            
    # Train the model
    def train_model(self):            
        # Ensure there is enough data in the memory to train
        if len(self.memory) < self.train_start:
            return
        
        # Sample a random batch of experiences from memory
        sample = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        # Initalize the states, actions, next_states, and done arrays
        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, rewards, done = [], [], []

        # Loop through the sample and append the states, actions, rewards, and done arrays
        for i in range(self.batch_size):
            # sample[i] is a tuple of (state, action, reward, next_state, done), here we are only appending it to the corresponding arrays
            state[i] = sample[i][0]
            action.append(sample[i][1])
            rewards.append(sample[i][2])
            next_state[i] = sample[i][3]
            done.append(sample[i][4])
            
        # Calculate the target Q values
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
            
        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = rewards[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = rewards[i] + self.gamma * (np.amax(target_next[i]))
        
        # train the model
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
        
    def run(self):            
        for eps in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            score = 0
            while not done:
                # gets the action from the model, either it is random or an actual prediction from the model depending on the epsilon value
                action = self.act(state)
                # shifts the game forward by one step
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done:
                    # if game is not done or if it is the last step of the game, then append the reward to the memory
                    reward = reward
                else:
                    # penalty for losing
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                score += reward
                if done: 
                    print("episode: {}/{}, score: {}, epsilon: {}".format(eps, self.EPISODES, score, self.epsilon))
                self.train_model()
                    
if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()