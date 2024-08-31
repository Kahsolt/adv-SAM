# Adversarial Attack over SAM with Various Loss Functions

    Explore PGD attack with various loss function over the Segment-Anything Model

----

### Quickstart

⚪ install

- run `init_repos.cmd` to init repos and download basic SAM checkpoints
  - run `init_repos.cmd all` if you need **all** SAM checkpoints
- run `pip install -r requirements.txt`

⚪ run

- run `run_gui.py` for interactive SAM app
  - use `-K <SAM|MobileSAM>` to change SAM backend :)
- run `atk_sam.py` for attack
- see cmdline exmpales in `run.cmd`


### Dataset

- segment-anything dataset: [https://ai.meta.com/datasets/segment-anything/](https://ai.meta.com/datasets/segment-anything/)
- unzip the under folder `data/SAM_data`, should contain 11186 examples

```
data/
  SAM_data/       # SAM dataset
    sa_*.png
    sa_*.json
```


#### reference

- Attack-SAM: [https://arxiv.org/abs/2305.00866](https://arxiv.org/abs/2305.00866)
- SegPGD: [https://arxiv.org/abs/2207.12391](https://arxiv.org/abs/2207.12391)

If you find this repo useful, please star and cite us:

```
@misc{kahsolt2023,
  author = {Kahsolt},
  title  = {Adversarial Attack over SAM with Various Loss Functions},
  howpublished = {\url{https://github.com/Kahsolt/adv-SAM}}
  month  = {June},
  year   = {2023}
}
```

----
by Armit
2023/06/23 
