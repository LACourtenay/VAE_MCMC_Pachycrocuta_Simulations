# Variational Autoencoders and Markov Chain Monte Carlo algorithms for the modelling of <i>Pachycrocuta brevirostris</i> tooth pits
Code for the training and implementation of Variational Autoencoders and Markov Chain Monte Carlo algorithms for the modelling and simulation of <i>Pachycrocuta brevirostris</i> tooth marks

-----------------------------------------------------------------------------------------------------------------

## <b> Author Details </b>

<b> Author </b>: Lloyd A. Courtenay

<b> Email </b>: ladc1995@gmail.com

<b> ORCID </b>: https://orcid.org/0000-0002-4810-2001

<b> Current Afiliation </b>: University of Bordeaux [CNRS, PACEA UMR5199]

---------------------------------------------------------------------------------------------------

This code has been designed for the open-source free R and Python programming languages.

---------------------------------------------------------------------------------------------------

## <b> Repository Details </b>

The present repository contains:

* <b> Supplementary File 1.pdf </b>
  * Supplementary text, figures and tables from the original paper
* <b> LICENSE </b>, <b>Courtenay_et_al_2021b_data.txt</b> and the <b>Rproj</b> files
  * The LICENSE file, additional data, and files used to launch and prepare the R Studio project environment
* <b> Pretrained VAE Weights </b>
  * Each of the h5 files containing trained VAE weights that were used for the purpose of this study
* <b> Code </b>
  * <b>MCMC Model.py</b>
    * Python code to load the trained VAE model, encode the target pachycrocuta tooth pits in latent space, and then use and MCMC algorithm to sample from the latent distribution. Afterwards these sampled latent coordiantes are used as input to the generator to simulate new tooth pits.
  * <b>R Code - GMM data.R</b>
    * R Code to load the landmark coordinates, superimpose them with the coordiantes of tooth pits left by Pachycrocuta brevirostris on the femur of a Hippo from FN3, and then export these superimposed Procrustes form coordiantes for modelling with the python code
    * This R Code also has a simple number of lines to load the simulated data and extract length, width and depth values from these simualted pits.
    * The R Code contains the landmark coordiantes obtained from the Hippo femur, but we do not have permission to share all of the reference landmark coordinates of modern carnivore species, because they come from other publications. The dataset from Courtenay et al. (2021) "Developments in data science solutions for carnivore tooth pit classification" is open access and has already been published. The dataset from Courtenay et al. (2021) "3D insigths into the effects of captivity on wolf mastication and their tooth marks; implications in ecological studies of both the past and present" is no longer associated to the original publication. To avoid issues with people interested in accessing this dataset, we have obtained permission to include this file in the present repository with the name <b>Courtenay_et_al_2021b_data.txt</b>. Our R code, however, has been adapted to load the data from each of these repositories, save them to a local disk, and then load them in the associated code.
  * <b>R Code - analyse python output.R</b>
    * R Code used to analyse the output of the VAE and MCMC models trained and executed in python. The final produced models from these files are the lengths, widths and depths of simulated tooth pits
  * <b>VAE Model.py</b>
      * Python code that was originally used to define and train the Variational Autoencoder. Weights from the trained encoder are included in the Trained VAE Weights folder.

--------------------------------------------------------

## <b> System Requirements </b>

All code was run in an Anaconda (v.23.3.1) environment

* Python
  * Python library version 3.10.9
  * Tensorflow
    * Python library version 2.12.0
  * Numpy
    * Python library version 1.22.0
* R
  * Version 4.3.0
  * geomorph
    * R library version 4.0.5
  * abind
    * R library version 1.4.0
  * GraphGMM
    * R library version 1.0
      * Available from https://github.com/LACourtenay/GraphGMM
  * ggplot2 
    * R library version 3.4.3
  * httr
    * R library version 1.4.7

# <b> Running the code </b>

Files need to be run in the following order: <b>R Code - GMM data.R</b>, <b>VAE Model.py</b>, <b>MCMC Model.py</b> and finally <b>R Code - analyse python output.R</b>. Our suggestion is to open R Studio (we used R Stuido version 2023.03.0, Build 386) using the <b>VAE_MCMC_Pachycrocuta_Simulations.Rproj</b> file to open an R Studio project. Once this has been done, the user can then simply source the <b>R Code - GMM data.R</b> file to perform the procrustes superimposition of all raw coordinate data, and create a new folder called data where data will begin to be stored. The user can then open a command prompt using cmd within the project folder, activate an anaconda environment, and launch both python codes. In our computer, we used an anaconda environment called tensorflow, containing all the associated versions of the python libraries installed. Once these two files have been run, in R Studio the user can then source the <b>R Code - analyse python output.R</b> to load into memory the simulated lengths, widths and depths of pachycrocuta individuals based on the data produced by both the VAE and MCMC algorithms in python.

--------------------------------------------------------

## <b> Citation </b>

Please cite this repository as:

 <b> Courtenay, L.A. (2024) Code and Data for the Modelling of Pachycrocuta tooth pits using Variational Autoencoders and Markov Chain Monte Carlo algorithms. https://github.com/LACourtenay/VAE_MCMC_Pachycrocuta_Simulations </b>

--------------------------------------------------------

Comments, questions, doubts, suggestions and corrections can all be directed to L. A. Courtenay at the email provided above.

---------------------------------------------------------------------------------------------------

## License

This project is licensed under the GNU Affero General Public License v3.0.
See the LICENSE file for details.

<b> Copyright (C) 2025 Lloyd Courtenay </b>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, version 3.

