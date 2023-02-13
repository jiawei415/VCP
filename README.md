# Value Consistency Prioritization (VCP)
Code for "*Efficient Multi-Goal Reinforcement Learning via Value Consistency Prioritization*". 


## Requirements
python3.6+, tensorflow, gym, mujoco

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/jiawei415/VCP.git
    cd VCP
    ```

- Install vcp package
    ```bash
    pip install -e .
    ```


## Usage
Environments: PointMassEmptyEnv-v1, Reacher-v2, FetchReach-v1, HandReach-v0, HandManipulatePenRotate-v0. 

VCP:
```bash
python -m  vcp.run --env PointMassEmptyEnv-v1 --num_epoch 50 --num_env 1  --alg_config "{'k_heads':16,'priority_temperature':9.0}
```

HER:
```bash
python -m  vcp.run --env PointMassEmptyEnv-v1 --num_epoch 50 --num_env 16  --alg_config "{'k_heads':1,'prioritized_replay':False,'use_her_buffer':False}"
```


