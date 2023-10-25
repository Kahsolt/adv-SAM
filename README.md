# Call White Black: Adversarial Attack over SAM

    Explore targeted PGD adversarial attack over the Segment-Anything Model

----

### **The official implementation of Call White Black: Mislead the Segment-Anything Model with Targeted Attack**  

Do not worry, our paper is on the way :) 🎉🎉🎉  


### Quickstart

⚪ install

- run `init_repos.cmd` to init repos and download basic SAM checkpoints
  - run `init_repos.cmd all` if you need **all** SAM checkpoints
- run `pip install -r requirements.txt`

⚪ run

- run `run_ptor_gui.py` for interactive SAM app
- run `atk_sam.py` for attack
- see cmdline exmpales in `run.cmd`


### Experiments

⚪ non-targeted

ℹ It is technically a simple flat-color target valued `loss_w`

| loss\meth | FGSM | PGD | SegPGD |
| :-: | :-: | :-: | :-: |
| MAE     |     |     |     |
| MSE     |     |     |     |
| ClipMAE |     |     |     |
| ClipMSE |     |     |     |
| BCE     |     |     |     |

⚪ targetd

| loss\meth | FGSM | PGD | SegPGD |
| :-: | :-: | :-: | :-: |
| MAE     |     |     |     |
| MSE     |     |     |     |
| ClipMAE |     |     |     |
| ClipMSE |     |     |     |
| BCE     |     |     |     |


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


#### credit

If you find this useful, please cite us:

```
@misc{kahsolt2023,
  author = {Kahsolt},
  title  = {Call White Black: Mislead the Segment-Anything Model with Targeted Attack},
  howpublished = {\url{https://github.com/Kahsolt/adv-SAM}}
  month  = {June},
  year   = {2023}
}
```

----
by Armit
2023/06/23 
