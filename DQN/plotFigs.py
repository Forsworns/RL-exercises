import matplotlib.pyplot  as plt
import json

rewards_file = 'training_rewards.json'
losses_file = 'training_losses.json'

with open(rewards_file,"r") as f:
    rewards = json.loads(f.read())

with open(losses_file,"r") as f:
    losses = json.loads(f.read())

plt.figure()
plt.plot(list(range(len(rewards))),rewards)
plt.xlabel("episode")
plt.ylabel("reward")
plt.show()

plt.figure()
plt.plot(list(range(len(rewards))),rewards)
plt.xlabel("episode")
plt.ylabel("loss")
plt.show()