# ASAP: Attention-Based State Space Abstraction for Policy Summarization

This repository contains code, data, models and Tensorboard log files for the abovementioned paper.

## Information

To reproduce the results for ASAP, as shown in the paper, run the Jupyter notebook named "results.ipynb". The repository includes the models used in the experiments and their Tensorboard log files showing the training. You do not need to retrain yourself to reproduce the results. They are stored in the folder named "runs".

To train ASAP from scratch, use the file "cluster_importance.py." The "requirements.txt" was made later, so it does not contain identical versions of packages as I initially used. Still, they produce the same results as far as I can tell.

The hyperparameters used to train ASAP are stored in the folder "configs". The repository includes the data, but since they are generated from simulations, it is possible to delete and regenerate them with this code.

## Acknowledgements

* The use of state space abstraction as an explanation is inspired by Topin and Veloso (2019) and McCalmon et al. (2022).
* The use of feature importance in explaining reinforcement learning is inspired by Greydanus et al. (2018) and Shi et al. (2022). The idea of learning to generate attention maps was inspired by Shi et al. (2022). The paper should have made this more explicit by expanding on the text "ASAP aims to complement the weakness of the aforementioned local explanation methods". The "aforementioned local explanation methods" in the paper cite these two methods and some other feature importance methods.
* The project was only possible with the many open-source packages and code. The licenses for the other open-source codes can be found in the file `LICENSE`.

All references mentioned are found in the paper.

References

* Joe McCalmon, Thai Le, Sarra Alqahtani, and Dongwon Lee. CAPS: Comprehensible Abstract Policy Summaries for Explaining Reinforcement Learning Agents. In Proc. of AAMAS, 2022. doi: 10.5555/3535850.3535950.
* Topin, N., & Veloso, M. (2019). Generation of policy-level explanations for reinforcement learning. In Proc. of AAAI, 2019. doi: 10.1609/aaai.v33i01.33012514.
* Nicholay Topin and Manuela Veloso. Generation of Policy-Level Explanations for Reinforcement Learning. In Proc. of AAAI, 2019. doi: 10.1609/aaai.v33i01.33012514.
* Wenjie Shi, Gao Huang, Shiji Song, Zhuoyuan Wang, Tingyu Lin, and Cheng Wu. Self-Supervised Discovering of Interpretable Features for Reinforcement Learning. IEEE Trans. Pattern Anal. Mach. Intell., 2022. doi: 10.1109/TPAMI.2020.3037898.

## Citation

```lang-tex
@inproceedings{DBLP:conf/acml/BekkemoenL23,
  author       = {Yanzhe Bekkemoen and
                  Helge Langseth},
  title        = {{ASAP:} Attention-Based State Space Abstraction for Policy Summarization},
  booktitle    = {In Proc. of ACML},
  year         = {2023},
  url          = {https://proceedings.mlr.press/v222/bekkemoen24a.html},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
