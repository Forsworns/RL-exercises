## Small Gridworld (Policy Evaluation & Improvement )

### 1. Files

The global constants are in the `config.py`

The gridworld environment is implemented in the `GridWorld.py`.

The policy evaluation is implemented in the `PE.py`.

The policy improvement is implemented in the `PI.py`

### 2. Results

The results are shown as follows:

Input

```shell
python ./PE.py
```

in a shell and we can get these outputs:

![PE](.\figs\PE.png)

![PEs](.\figs\PEs.png)

Then input

```python
python ./PI.py
```

We will get

![PI](.\figs\PI.png)

![PIs](.\figs\PIs.png)

We can also assign other arguments to verify the generalization of these two algorithms, which will be discussed in the next part.

### 3. Details

The environment and two algorithms are  packaged into three classes. The `argparse ` library is used to parse the arguments from the terminal. These arguments can also be referred direcly when we create a new object instance by construction function.

The supplied argument are as follows:

For GridWorld

| argument |             help information             | default |
| :------: | :--------------------------------------: | :-----: |
| --gamma  |              the grid size               |    4    |
| --target | the discount for future expected rewards |    1    |
|  --grid  |           the target position            |   []    |
| --reward |            the future reward             |   -1    |

For PolicyEvaluation and PolicyImprovement

| argument |                   help information                   | default |
| :------: | :--------------------------------------------------: | :-----: |
| --delta  | the threshold to stop the iteration (only for PE.py) |  0.001  |
|  --ite   |                  maximum iterations                  |   400   |

For example, we can use `python PE.py --delta=0 `

![PE_](.\figs\PE_.png)