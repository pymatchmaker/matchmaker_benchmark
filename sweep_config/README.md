## General Use Case:

* create sweep
```
wandb sweep --entity matchmaker --project <project_name> <config-file_path>
```

* run agent
```
wandb agent <entity_name>/<project_name>/<sweep_id>
```

---------------
### Example:
```
wandb sweep --entity matchmaker --project hmm-hyperparameters sweep_config/hmm-minimalistic.yaml
```

-> ` wandb: Creating sweep with ID: 86sclfcl `

```
wandb agent matchmaker/hmm-hyperparameters/86sclfcl
```

--------------
### Docs:
Here are multiple ways how to do this hyperparameter search and how to set the sweep_config:

[https://docs.wandb.ai/models/sweeps/sweep-config-keys](https://docs.wandb.ai/models/sweeps/sweep-config-keys)