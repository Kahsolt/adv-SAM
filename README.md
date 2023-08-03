# adv-SAM

    Edge-PGD adversarial attack over the Segment-Anything model

----

### Quickstart

⚪ install

- run `init_repos.cmd` to init repos and download basic SAM checkpoints
  - run `init_repos.cmd all` if you need **all** SAM checkpoints
- run `pip install -r requirements.txt`

⚪ run

- run `run_ptor_gui.py` for interactive SAM app
- run `atk_sam.py` for PGD and Edge-PGD attack
  - see cmdline exmpales in `run.cmd`
- run `atk_sam_dataset.py` to test over the dataset


### Dataset

- segment-anything dataset: [https://ai.meta.com/datasets/segment-anything/](https://ai.meta.com/datasets/segment-anything/)
- unzip the under folder `data/SAM_data`, should contain 11186 examples

```
data/
  SAM_data/
    sa_*.png
    sa_*.json
```

----
by Armit
2023/06/23 
