# PI-DRME
"Physics-Inspired Distributed Radio Map Estimation" 2025 IEEE International Conference on Communications.

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
(3) Standalone learning: enter ./standalone_badedDistV2 folder, then perform
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


