# Point Cloud Diffusion for Automatic Reconstruction

(This README is still a work in progress)

This is the repository of the Memoir work of Pablo Jaramillo for qualifying to the degree of Civil Computer Engineer from University of Chile's (UCH) Department of Computer Sciences (DCC) at the Physical and Mathematical Sciencies Faculty (FCFM). It consists of a fork of the [work of Friedrich et al. For Automatic Implant Generation](https://github.com/pfriedri/pcdiff-implant). See the previous link for the original code, and [this link for the original project page](https://pfriedri.github.io/pcdiff-implant-io/).

## Nature of the work

This work agrees with the statements of its source for Diffusion Models to be a powerful architecture for faithful and high-resolution 3D reconstructions. It builds upon the previous work by proposing the general usage of the model in other fields, such as, archaeological reconstruction. The model is capable of reconstructing a variety of shapes but requires a low degree of diversity in the data, preferable in a single item-class dataset, where the reconstruction task is confined to either a single location within the object or a single type of surface. Examples of such applications are: the original purpose for cranial implant generation, and the challenged in this work - pottery bottom reconstruction.

## Model requisites

In order to run this model you will need at minimum a GPU with 8GB of memory, though more memory or more GPUs are required to run at higher point cloud density. For best performance it is recommended that your dataset has somewhere between 900 to 2000 data points, split in a 75-25 or 70-30 train-test ratio, should your dataset be smaller augmentation is suggested. The point count of the point clouds should be fixed at no lower than 4000. The convergence of the Point Cloud Diffusion Model may be expected within 400 to 500 epochs following this configuration.

## Acknowledgements

The neural network model code is authored by Friedrich et al. Published under a MIT License, this license is kept as-is without change from the original repository. Friedrich's helpful disposition was appreaciated when first starting the project.

Many thanks to Ivan Sipiran, supervisor of Pablo for the Memoir, and co-author of the paper published detailing the findings obtained.