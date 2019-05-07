## MountainCar(DQN) 

### 1. Files

This experiment is based on the MountrainCar-v0 environment in the `env.py`.

The global configurations are in the `configs.py`.

The DQN algorithm is implemented in the `agent.py`.

The entry is in the `main.py`.

### 2. Results

The algorithm is implemented in tensorflow and it can converge in 500 steps for MountainCar-v0 environment. The optimizer is AdamOptimizer and the learning rate is set  to 0.001. The discount is set to constant 1 as the paper recommends.  Batch size is 32 and the memory is 5000. 

To test the model, use

```shell
python main.py
```

You could read the reward and learn whether the car arrive the top of the mountain from the shell line as the following figure shows.

![](.\figs\test.png)

To train a model, use

```bash
python main.py --is_train=True
```

Here are some screenshots of the training process

![](.\figs\train_1.png)

![](.\figs\train_2.png)

The reward is plotted as follows

![](.\figs\reward.png) 

### 3. Details

We can use the arguments to change the configuration of the model. Here are the arguments in the `main.py`

|    argument     |                help information                | default |
| :-------------: | :--------------------------------------------: | :-----: |
|    --use_gpu    |           Whether to use gpu or not            |  True   |
|    --display    |      Whether to do display the env or not      |  True   |
|   --is_train    |                 Train or test                  |  True   |
| --gpu_fraction  |            idx / # of gpu fraction             |   1/1   |
|  --tagt_q_step  |   Steps interval to update target q network    |         |
|    --epsilon    |     Probability to take random exploration     |    1    |
|  --queue_size   | The memory size to implement experience replay |   500   |
| --enable_replay |        Whether to use experience replay        |  True   |
| --enable_target |         Whether to use target network          |  True   |

We record three parameters during experiment. They are `enable_target`, `tagt_q_step`,  and `queue_size`. The trained model are saved under the corresponding directory.

We implements two kinds of DQN. The first is the NIPS-DQN published in 2013. The second is the well-known Nature-DQN in 2015. The only different is the update of the q-value. The NIPS-DQN updates q with the same network as the following equation

$y_{j}=\left\{\begin{array}{ll}{r_{j}} & {\text { for terminal } \phi_{j+1}} \\ {r_{j}+\gamma \max _{a^{\prime}} Q\left(\phi_{j+1}, a^{\prime} ; \theta\right)} & {\text { for non-terminal } \phi_{j+1}}\end{array}\right.,$

while the Nature-DQN updates with a similar network updated asynchronously:

$y_{j}=\left\{\begin{array}{ll}{r_{j}} & {\text { for terminal } \phi_{j+1}} \\ {r_{j}+\gamma \max _{a^{\prime}} Q\left(\phi_{j+1}, a^{\prime} ; \theta\right)} & {\text { for non-terminal } \phi_{j+1}}\end{array}\right..$

We also add an argument to control the memory replay, as you can see in the table. And the result without memory replay is not ideal.

We compare the NIPS-DQN and Nature-DQN and find that our Nature-DQN outperforms NIPS-DQN. So the default network parameters for testing are fitted by Nature-DQN. This is consistent with the theoretical conclusion.

We find the reward setting is one of the most important part in the algorithm. But it's really hard to determine the formula of the reward. We set the reward to $(position-(-0.6))*10$. Because we find that the lowest position is -0.6 in the source code of the MountainCar-v0.  We multiply it by 10 to magnify the reward. To emphasize our goal, reaching the summit, we add 200 to the reward whenever the car reach a height higher than the summit. The height of the summit is 0.5 according to the source code of MountainCar-v0 environment.

Another trick is the exploration process. We find DQN hard to converge or easy to converge to a local minima if we don't explore enough times. So we explore a lot at the beginning and slowly decay the exploration rate.

We also find that the network structure has an effect on the performance of DQN. We tried a network structure with only one hidden layer, and we can't get a satisfying result. Therefore, in the end, we use two hidden layers in our network. 