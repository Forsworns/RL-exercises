import matplotlib.pyplot  as plt
import json

rewards_file = 'training_rewards.json'

with open(rewards_file,"r") as f:
    rewards = json.loads(f.read())

plt.figure()
plt.plot(list(range(len(rewards))),rewards)
plt.xlabel("episode")
plt.ylabel("reward")
plt.show()
