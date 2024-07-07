
.rs.restartR() # restart the R session in R studio

# libraries and functions ---------------------------------------------

library(geomorph)
library(ggplot2)
library(GraphGMM)

euclid <- function(point1, point2) {
  return(sqrt(sum((point1 - point2)^2)))
}

depth_estimate <- function(point1, point2, point3) {
  
  a <- euclid(point1, point2)
  b <- euclid(point2, point3)
  c <- euclid(point3, point1)
  
  s <- (a + b + c) / 2
  area <- sqrt(s * (s - a) * (s - b) * (s - c))
  height <- (2 * area) / a
  
  return(height)
  
}

# Analyse output of VAE and MCMC --------------------------------------------------------------

simulated_pachycrocuta1 <- read.table(".\\data\\reconstructed_pachycrocuta1.txt", header = FALSE, sep = ",")
simulated_pachycrocuta2 <- read.table(".\\data\\reconstructed_pachycrocuta2.txt", header = FALSE, sep = ",")
simulated_pachycrocuta3 <- read.table(".\\data\\reconstructed_pachycrocuta3.txt", header = FALSE, sep = ",")
simulated_pachycrocuta4 <- read.table(".\\data\\reconstructed_pachycrocuta4.txt", header = FALSE, sep = ",")
simulated_pachycrocuta <- rbind(simulated_pachycrocuta1, simulated_pachycrocuta2, simulated_pachycrocuta3, simulated_pachycrocuta4)
np1 <- nrow(simulated_pachycrocuta)
simulated_pachycrocuta_tensor <- array(numeric(), dim = c(25, 3, 0))
for (i in 1:np1) {
  
  landmarks_prima <- matrix(simulated_pachycrocuta[i,], nrow = 25, byrow = TRUE)
  simulated_pachycrocuta_tensor <- abind::abind(simulated_pachycrocuta_tensor, landmarks_prima, along = 3)
  
}

# calculated simulated properties of tooth pits 

sim_pachy_length <- c()
sim_pachy_width <- c()
sim_pachy_depth <- c()

for (i in 1:dim(simulated_pachycrocuta_tensor)[3]) {
  
  sim_pachy_length <- c(sim_pachy_length, euclid(simulated_pachycrocuta_tensor[1,,i],
                                                   simulated_pachycrocuta_tensor[2,,i]))
  sim_pachy_width <- c(sim_pachy_width, euclid(simulated_pachycrocuta_tensor[3,,i],
                                                 simulated_pachycrocuta_tensor[4,,i]))
  sim_pachy_depth <- c(sim_pachy_depth, depth_estimate(simulated_pachycrocuta_tensor[1,,i],
                                                         simulated_pachycrocuta_tensor[2,,i],
                                                         simulated_pachycrocuta_tensor[5,,i]))
  
}

