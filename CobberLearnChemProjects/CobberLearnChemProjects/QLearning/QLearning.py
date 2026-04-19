# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 2. CREATE ENVIRONMENT
# =========================================================
env = gym.make("FrozenLake-v1", is_slippery=False)

# Print observation and action space (REQUIRED)
print("Observation space (states):", env.observation_space)
print("Action space:", env.action_space)

n_states = env.observation_space.n   # 16
n_actions = env.action_space.n       # 4 (LEFT, DOWN, RIGHT, UP)

# =========================================================
# 3. CREATE Q-TABLE (BRAIN)
# =========================================================
Q = np.zeros((n_states, n_actions))  # agent starts with no knowledge

# =========================================================
# 4. TRAINING PARAMETERS
# =========================================================
alpha = 0.8
gamma = 0.95

epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.1

episodes = 20000
max_steps = 100

rewards_per_episode = []

# =========================================================
# 5. LEARNING LOOP (FOR + WHILE REQUIRED)
# =========================================================
for episode in range(episodes):

    state, _ = env.reset()
    done = False
    step = 0
    total_reward = 0

    while not done and step < max_steps:

        # -----------------------------------------
        # ACTION SELECTION (exploration vs exploit)
        # -----------------------------------------
        if np.random.rand() < epsilon:
            action = env.action_space.sample()   # random action (exploration)
        else:
            action = np.argmax(Q[state])         # best known action

        # -----------------------------------------
        # ENVIRONMENT STEP
        # -----------------------------------------
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # -----------------------------------------
        # BELLMAN EQUATION UPDATE
        # -----------------------------------------
        best_next = np.max(Q[new_state])
        target = reward + gamma * best_next * (1 - done)

        Q[state, action] += alpha * (target - Q[state, action])

        state = new_state
        total_reward += reward
        step += 1

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

# =========================================================
# 6. PRINT FINAL Q-TABLE
# =========================================================
np.set_printoptions(precision=2)
print("\nFINAL Q-TABLE:\n", Q)

# =========================================================
# 7. POLICY INTERPRETATION (ARROWS)
# =========================================================
arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}

def print_policy(Q):
    policy = np.argmax(Q, axis=1).reshape(4, 4)

    print("\nLEARNED POLICY:")
    for row in policy:
        print(" ".join(arrows[action] for action in row))

print_policy(Q)

# =========================================================
# 8. TEST AGENT (100 EPISODES, NO EXPLORATION)
# =========================================================
test_episodes = 100
success = 0

for _ in range(test_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(Q[state])  # greedy policy
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if reward == 1:
            success += 1
            break

success_rate = (success / test_episodes) * 100
print("\nSUCCESS RATE:", success_rate, "%")

# =========================================================
# 9. LEARNING CURVE VISUALIZATION
# =========================================================
window = 200
moving_avg = np.convolve(
    rewards_per_episode,
    np.ones(window) / window,
    mode='valid'
)

plt.plot(moving_avg)
plt.title("Learning Curve (Average Reward Over Time)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()