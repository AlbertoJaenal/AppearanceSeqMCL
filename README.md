# AppearanceSeqMCL
This repository corresponds to the work entitled "[Unsupervised appearance map abstraction for indoor Visual Place Recognition with mobile robots](https://ieeexplore.ieee.org/abstract/document/9808108/)", published at IEEE Robotics and Automation Letters. 


**Authors:** [Alberto Jaenal](https://mapir.isa.uma.es/mapirwebsite/?p=2022), [Francisco-Angel Moreno](https://mapir.isa.uma.es/mapirwebsite/?p=1721) and [Javier Gonzalez-Jimenez](https://mapir.isa.uma.es/mapirwebsite/?p=1536)


**Video:** [Click Here](https://youtu.be/4vkuK4_RfVQ)

![](output/Top_GAM.png)![](output/Met_GAM.png)

## Cite
If you use this work in your research, please cite:

```
@article{jaenal2023sequential,
  title={Sequential Monte Carlo localization in topometric appearance maps},
  author={Jaenal, Alberto and Moreno, Francisco-Angel and Gonzalez-Jimenez, Javier},
  journal={The International Journal of Robotics Research},
  pages={02783649231197723},
  year={2023},
  publisher={SAGE Publications Sage UK: London, England}
}
```

## Instructions

1. Check the jupyter called Appearance-based Localization with Local Observation Models.ipynb.

Optional. There is an available implementation of the [Gaussian Process Particle Filter](https://ieeexplore.ieee.org/abstract/document/7743697)

## Dependencies

This software employs built-in libs (see `requeriments.txt`), and has been tested with Python>=3.5 on Ubuntu 16.04, 18.04 and 20.04.

The `geometry.py` script is inspired in [ProbFiltersVPR](https://github.com/mingu6/ProbFiltersVPR).
This repo reuses the Expectation-Maximization algorithm for Topological GAM generation, which is also available [here](https://github.com/AlbertoJaenal/MapAbstractionVPR).