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

* <b> Trained VAE Weights </b>
  * Each of the h5 files containing trained VAE weights that were used for the purpose of this study
* <b> Code </b>
  * <b>MCMC Model.py</b>
    * Python code to load the trained VAE model, encode the target pachycrocuta tooth pits in latent space, and then use and MCMC algorithm to sample from the latent distribution. Afterwards these sampled latent coordiantes are used as input to the generator to simulate new tooth pits.
  * <b>R Code - GMM data.R</b>
    * R Code to load the landmark coordinates, superimpose them with the coordiantes of tooth pits left by Pachycrocuta brevirostris on the femur of a Hippo from FN3, and then export these superimposed Procrustes form coordiantes for modelling with the python code
    * This R Code also has a simple number of lines to load the simulated data and extract length, width and depth values from these simualted pits.
    * The R Code contains the landmark coordiantes obtained from the Hippo femur, but we do not have permission to share the reference landmark coordinates of modern carnivore species, because they come from other publications. These coordinates however are open access and have already been published. Consult the original publications for details on how to access them.
  * <b>VAE Model.py</b>
      * Python code that was originally used to define and train the Variational Autoencoder. Weights from the trained encoder are included in the Trained VAE Weights folder.

Files need to be run in the following order: <b>R Code - GMM data.R</b>, <b>VAE Model.py</b>, <b>MCMC Model.py</b> and finally <b>R Code - analyse python output.R</b>.

--------------------------------------------------------

## <b> System Requirements for Deep Learning </b>

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

--------------------------------------------------------

## <b> Citation </b>

Please cite this repository as:

 <b> Courtenay, L.A. (2024) Code and Data for the Modelling of Pachycrocuta tooth pits using Variational Autoencoders and Markov Chain Monte Carlo algorithms. https://github.com/LACourtenay/VAE_MCMC_Pachycrocuta_Simulations </b>

--------------------------------------------------------

Comments, questions, doubts, suggestions and corrections can all be directed to L. A. Courtenay at the email provided above.
