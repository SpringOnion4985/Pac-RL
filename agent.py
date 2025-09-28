import random
import pickle
import os

class QLearningAgent:
    ## CHANGED: Added parameters for epsilon decay for better training.
    def __init__(self, actions, epsilon=1.0, alpha=0.1, gamma=0.9, epsilon_decay=0.9995, min_epsilon=0.01):
        self.actions = actions
        self.alpha = alpha              # Learning rate
        self.gamma = gamma              # Discount factor
        self.epsilon = epsilon          # Exploration rate
        self.epsilon_decay = epsilon_decay # Rate at which epsilon decays
        self.min_epsilon = min_epsilon   # Minimum value for epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        """Get q-value from table, return 0 if it's not there."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """Decide whether to explore (random move) or exploit (best-known move)."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            q_values = [self.get_q_value(state, a) for a in self.actions]
            max_q = max(q_values)
            
            # In case of a tie, randomly choose one of the best actions
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return self.actions[random.choice(best_actions)]  # Exploit

    def learn(self, state, action, reward, next_state):
        """Update q-table with new information using the Q-learning formula."""
        old_q = self.get_q_value(state, action)
        future_q = max([self.get_q_value(next_state, a) for a in self.actions])
        
        # The Q-learning formula
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.q_table[(state, action)] = new_q

    ## CHANGED: New method to save the learned Q-table to a file.
    def save_q_table(self, filename):
        """Saves the Q-table to a file using pickle."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
            print(f"Q-table successfully saved to {filename}")
        except Exception as e:
            print(f"Error saving Q-table: {e}")

    ## CHANGED: New method to load a pre-existing Q-table.
    def load_q_table(self, filename):
        """Loads the Q-table from a file if it exists."""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    self.q_table = pickle.load(f)
                print(f"Q-table loaded from {filename}")
            except Exception as e:
                print(f"Error loading Q-table: {e}. Starting fresh.")
        else:
            print(f"No Q-table found at {filename}, starting with a new one.")