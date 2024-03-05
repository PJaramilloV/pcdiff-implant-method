# Point Cloud Diffusion Models for Automatic Implant Generation
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-page-blue)](https://pfriedri.github.io/pcdiff-implant-io/)
[![arXiv](https://img.shields.io/badge/arXiv-2303.08061-b31b1b.svg)](https://arxiv.org/abs/2303.08061)

This is the official PyTorch implementation of the MICCAI 2023 paper [Point Cloud Diffusion Models for Automatic Implant Generation](https://pfriedri.github.io/pcdiff-implant-io/) by Paul Friedrich, Julia Wolleb, Florentin Bieder, Florian M. Thieringer and Philippe C. Cattin.

If you find our work useful, please consider to :star: **star this repository** and :memo: **cite our paper**:
```bibtex
@InProceedings{10.1007/978-3-031-43996-4_11,
                author="Friedrich, Paul and Wolleb, Julia and Bieder, Florentin and Thieringer, Florian M. and Cattin, Philippe C.",
                title="Point Cloud Diffusion Models for Automatic Implant Generation",
                booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
                year="2023",
                pages="112--122",
               }
```


## Paper Abstract
Advances in 3D printing of biocompatible materials make patient-specific implants increasingly popular. The design of these implants is, however, still a tedious and largely manual process. Existing approaches to automate implant generation are mainly based on 3D U-Net architectures on downsampled or patch-wise data, which can result in a loss of detail or contextual information. Following the recent success of Diffusion Probabilistic Models, we propose a novel approach for implant generation based on a combination of 3D point cloud diffusion models and voxelization networks. Due to the stochastic sampling process in our diffusion model, we can propose an ensemble of different implants per defect from which the physicians can choose the most suitable one. We evaluate our method on the SkullBreak and SkullFix dataset, generating high-quality implants and achieving competitive evaluation scores.

![](./media/overview_pipeline.png)

## Data
We trained our network on the publicly available parts of the [SkullBreak/SkullFix](https://www.sciencedirect.com/science/article/pii/S2352340921001864) datasets.
The data is available at:
* SkullBreak: https://www.fit.vutbr.cz/~ikodym/skullbreak_training.zip
* SkullFix: https://files.icg.tugraz.at/f/2c5f458e781a42c6a916/?dl=1 (we just use the data in ```training_set.zip```)

The provided code works for the following data structure:
```
datasets
└───SkullBreak
    └───complete_skull
    └───defective_skull
        └───bilateral
        └───frontoorbital
        └───parietotemporal
        └───random_1
        └───random_2   
    └───implant
        └───bilateral
        └───frontoorbital
        └───parietotemporal
        └───random_1
        └───random_2
└───SkullFix
    └───complete_skull
    └───defective_skull
    └───implant
```

## Training & Using the Networks
Both networks, the point cloud diffusion model and the voxelization network are trained independently:
* Information on training and using the point cloud diffusion model can be found [here](./pcdiff/README.md)
* Information on training and using the voxelization network can be found [here](./voxelization/README.md)

## Implementation Details for Comparing Methods
* **3D U-Net**:  For implementing the paper [Improving the Automatic Cranial Implant Design in Cranioplasty by Linking Different Datasets](https://link.springer.com/chapter/10.1007/978-3-030-92652-6_4), we used their publicly available [implementation](https://github.com/MWod/AutoImplant_2021). As described in the paper we trained for 1000 iteration with 500 cases per iteration. The batch size was set to 1 and the Adam optimizer with a learning rate of 0.002 was used. The learning rate was decreased by a factor of 0.997 per iteration. We trained the network without linking both datasets via registration.
* **3D U-Net (sparse)**: For implementing the paper [Sparse Convolutional Neural Network for Skull Reconstruction](https://link.springer.com/chapter/10.1007/978-3-030-92652-6_7), we used their publicly available [implementation](https://github.com/akroviakov/SparseSkullCompletion). As described in their paper, we trained for 16 epochs using a batch size of 1. As proposed in the paper, we used an Adam optimizer with a learning rate of 0.001.
* **2D U-Net**: For implementing the paper [Cranial Implant Prediction by Learning an Ensemble of Slice-Based Skull Completion Networks](https://link.springer.com/chapter/10.1007/978-3-030-92652-6_8), we used their publicly available [implementation](https://github.com/YouJianFengXue/Cranial-implant-prediction-by-learning-an-ensemble-of-slice-based-skull-completion-networks). As there was no implementation detail given in the paper, we just followed the comments in the jupyter notebook.

All experiments were performed on an NVIDIA A100 (40GB) GPU.

## Results
### Results on the SkullBreak dataset
In the following table, we present the achieved evaluation scores as mean values over the SkullBreak test set.
We evaluate the Dice Score (DSC), the 10 mm boundary Dice Score (bDSC) and the 95 percentile Hausdorff Distance (HD95).
An example implant is shown [here](/media/example_skullbreak.png).
| Method            |DSC |bDSC|HD95|
|-------------------|----|----|----|
| 3D U-Net          |0.87|0.91|2.32|
| 3D U-Net (sparse) |0.71|0.80|4.60|
| 2D U-Net          |0.87|0.89|2.13|
| **Ours**          |0.86|0.88|2.51|
| **Ours (n=5)**    |0.87|0.89|2.45|

### Results on the SkullFix dataset
In the following table, we present the achieved evaluation scores as mean values over the SkullFix test set.
We evaluate the Dice Score (DSC), the 10 mm boundary Dice Score (bDSC) and the 95 percentile Hausdorff Distance (HD95). An example implant is shown [here](/media/example_skullfix.png).
| Method            |DSC |bDSC|HD95|
|-------------------|----|----|----|
| 3D U-Net          |0.91|0.95|1.79|
| 3D U-Net (sparse) |0.81|0.87|3.04|
| 2D U-Net          |0.89|0.92|1.98|
| **Ours**          |0.90|0.92|1.73|
| **Ours (n=5)**    |0.90|0.93|1.69|

### Runtime & GPU memory requirement information
In the following table, we present detailed runtime, as well as GPU memory requirement information. All values have been measured on a system with an AMD EPYC 7742 CPU and an NVIDIA A100 (40GB) GPU.

| Dataset (Method)                     |Point Cloud Diffusion Model | Voxelization Network | Total Time |
|--------------------------------------|----------------------------|----------------------|------------|
| SkullBreak (without ensembling, n=1) |~ 979 s, 4093 MB            |~ 23 s, 12999 MB      |~ 1002 s    |
| SkullBreak (with ensembling, n=5)    |~ 1101 s, 12093 MB          |~ 41 s, 12999 MB      |~ 1142 s    |
| SkullFix (without ensembling, n=1)   |~ 979 s, 4093 MB            |~ 92 s, 12999 MB      |~ 1071 s    |
| SkullFix (with ensembling, n=5)      |~ 1101 s, 12093 MB          |~ 109 s, 12999 MB     |~ 1210 s    |

Generating implants for the SkullFix dataset takes longer, as the volume output by the voxelization network (512 x 512 x 512) needs to be resampled to the initial volume size (512 x 512 x Z), which varies for different implants.

## Acknowledgements
Our code is based on/ inspired by the following repositories:
* https://github.com/autonomousvision/shape_as_points (published under [MIT license](https://github.com/autonomousvision/shape_as_points/blob/main/LICENSE))
* https://github.com/alexzhou907/PVD (published under [MIT license](https://github.com/alexzhou907/PVD/blob/main/LICENSE))

The code for computing the evaluation scores is based on:
* https://github.com/OldaKodym/evaluation_metrics
