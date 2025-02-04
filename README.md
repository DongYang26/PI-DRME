# PI-DRME
"Physics-Inspired Distributed Radio Map Estimation" 2025 IEEE International Conference on Communications. [Pdf](https://arxiv.org/pdf/2502.00319)

## Abstract
To gain panoramic awareness of spectrum coverage in complex wireless environments, data-driven learning approaches have recently been introduced for radio map estimation (RME). While existing deep learning based methods conduct RME given spectrum measurements gathered from dispersed sensors in the region of interest, they rely on centralized data at a fusion center, which however raises critical concerns on data privacy leakages and high communication overloads. Federated learning (FL) enhance data security and communication efficiency in RME by allowing multiple clients to collaborate in model training without directly sharing local data. However, the performance of the FL-based RME can be hindered by the problem of task heterogeneity across clients due to their unavailable or inaccurate landscaping information. To fill this gap, in this paper, we propose a physics-inspired distributed RME solution in the absence of landscaping information. The main idea is to develop a novel distributed RME framework empowered by leveraging the domain knowledge of radio propagation models, and by designing a new distributed learning approach that splits the entire RME model into two modules. A global autoencoder module is shared among clients to capture the common pathloss influence on radio propagation pattern, while a client-specific autoencoder module focuses on learning the individual features produced by local shadowing effects from the unique building distributions in local environment. Simulation results show that our proposed method outperforms the benchmarks in achieving higher performance.


## Dataset

This code is based on the [RadioMapSeer Dataset](https://radiomapseer.github.io/).

## Usage

### Requirements

- python
- pytorch
- pytorch-lightning = 1.9.4

## Run the following command:

(1) Distributed learning: enter ./Distributed_V2 folder, then perform
```
bash train_fl.sh
```
(2) Federated learning: enter ./Federated folder, then run
```
bash train.sh
```
(3) Standalone learning: enter ./standalone_basedDistV2 folder, then perform
```
bash train_standalone.sh
```

If you have any questions or suggestions regarding the code, please feel free to contact us:

Email: dnyang26@gmail.com

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

1. RME-GAN: https://github.com/achinthaw/RME-GAN
2. RadioUNet: https://github.com/RonLevie/RadioUNet


## Reference
Please cite the following paper if the codes is useful/inspiring for your research.

```
@INPROCEEDINGS{10758056,
  author={Dong Yang, Yue Wang, Songyang Zhang, Yingshu Li, Zhipeng Cai},
  booktitle={2025 IEEE International Conference on Communications}, 
  title={Physics-Inspired Distributed Radio Map Estimation}, 
  year={2025},
  pages={1-6}}
  ```


