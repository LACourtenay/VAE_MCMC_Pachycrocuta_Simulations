
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

#

# prepare procrustes landmark data ---------------------------------

# load the landmark dataset from a morphologika file

dataset <- read.morphologika("Carnivores.txt")
dataset$coords <- dataset$coords[c(1:7,9:15,17,19,21:27,29,30),,] # remove duplicate landmarks
dataset$labels <- as.factor(dataset$labels)

# introduce the four pachycrocuta tooth pits

pachycrocuta1 <- matrix(
  c(
    9.118104492,5.116993652,0.204683197,
    2.185940186,2.295411865,-0.028554634,
    4.230246582,6.450744629,0.12814978,
    6.748652832,0.974514099,0.483675049,
    6.221261719,3.633300781,-0.749728577,
    4.039686523,0.612284363,0.061832268,
    5.329230957,0.521267761,0.29147894,
    7.292429199,1.605414185,0.50913913,
    8.190063965,2.460361328,0.471369019,
    2.875614502,1.106271606,-0.057314743,
    4.743256836,1.809749146,-0.404657288,
    6.490893066,2.388288086,-0.295121216,
    7.669706055,3.17218209,-0.293327057,
    8.783539063,3.558662842,0.244224319,
    4.080813965,2.764800293,-0.469317932,
    7.821486328,4.462655273,-0.451463013,
    1.991216187,3.69067041,0.072025925,
    3.928077148,4.247489258,-0.416282715,
    5.172196289,5.022937012,-0.388980774,
    6.663042969,5.527063965,-0.552365601,
    8.14926416,6.157536621,0.210140381,
    2.589308594,4.950815918,0.262782562,
    3.55660791,5.871336426,0.287536316,
    5.432427246,6.597069824,-0.360137238,
    6.749741211,6.583394043,-0.597880066
  ),
  byrow = TRUE,
  nrow = 25
)
pachycrocuta2 <- matrix(
  c(
    3.533267822,8.463063477,0.307195587,
    8.709664063,10.49239258,0.146342484,
    7.711461914,7.755187988,0.217586166,
    5.557698242,11.81031784,0.36656601,
    7.06635791,9.977691406,-0.825625916,
    7.316629884,11.52299023,0.140800491,
    6.491405273,11.81516699,0.267570038,
    4.845291016,11.42086719,0.177302185,
    4.248304199,10.85027734,0.208554169,
    8.089294922,11.16771875,0.108945694,
    7.103161133,11.16262012,-0.101401299,
    6.15792041,11.30531152,-0.171463318,
    5.162413574,10.75337012,-0.089147682,
    3.714212646,9.858228516,0.289986603,
    8.090791504,10.29816406,-0.495013855,
    5.201919922,9.158483398,-0.472903137,
    8.834600586,9.613316406,0.178714874,
    8.099665039,9.200251953,-0.102776169,
    7.417236816,8.645092773,-0.477155151,
    6.434514648,8.369706055,-0.443530304,
    4.368295898,7.731550293,-0.048049725,
    8.835522461,8.629688477,0.348584839,
    8.30749141,8.087222656,0.249388718,
    6.944873535,7.57248584,-0.390134033,
    5.75257373,7.474611328,-0.517308044
  ),
  byrow = TRUE,
  nrow = 25
)
pachycrocuta3 <- matrix(
  c(
    3.936633301,3.337337158,0.03362904,
    2.254541992,2.395762451,-0.007312703,
    2.504732422,3.686996582,-0.140033112,
    3.566691406,2.099099365,0.032828991,
    3.158309082,2.787218506,-0.367510773,
    2.767477783,2.041079224,0.062773483,
    3.141960938,1.960571411,0.075939873,
    3.735312988,2.276453369,0.02895727,
    3.859499268,2.572261475,0.026870356,
    2.4367229,2.11155176,0.00537148,
    3.134682617,2.326993896,-0.207900925,
    3.357911377,2.433786377,-0.180766815,
    3.551245361,2.615547607,-0.161920624,
    3.924356445,2.901289551,0.009780197,
    2.696482178,2.562003662,-0.28824645,
    3.470008301,3.03428418,-0.262333862,
    2.10987793,2.651507324,-0.010515945,
    2.573616943,3.071932617,-0.312547424,
    2.770174075,3.350512939,-0.283020111,
    3.092209229,3.395182129,-0.236879517,
    3.702137695,3.675266846,0.089591423,
    2.136028564,3.050004883,-0.079149139,
    2.264064697,3.382102051,-0.117706497,
    2.870208252,3.812023926,-0.097271622,
    3.344593994,3.878933594,0.012260541
  ),
  byrow = TRUE,
  nrow = 25
)
pachycrocuta4 <- matrix(
  c(
    5.098495177,1.525454712,-0.167440582,
    5.177029785,8.840314453,-0.020862825,
    6.954594238,4.852897946,0.215004867,
    2.90570752,4.872006348,0.047191853,
    5.383678711,6.446483398,-0.720692566,
    3.485137695,7.570562012,0.051504948,
    2.97854126,6.32442041,0.035873928,
    3.025628174,3.326082031,0.133894318,
    3.438955322,2.144907959,0.008552785,
    4.23776123,8.383327148,0.012293341,
    4.200879883,7.027146484,-0.4444664,
    3.951783203,5.807149902,-0.382735199,
    4.238992188,4.254919922,-0.405717621,
    4.218314941,1.487360596,0.044121597,
    5.430318848,7.620668945,-0.491737762,
    5.142260254,4.419777832,-0.669707642,
    7.010275879,5.985669434,0.146087921,
    6.124302246,6.946803711,-0.048657593,
    6.203583008,5.829867188,-0.563259033,
    5.874711426,4.359353516,-0.651876404,
    5.858102539,1.6763125,-0.117239433,
    6.800019043,7.185766113,0.175850418,
    6.29185498,8.187234375,0.173799088,
    7.130877441,3.574295654,-0.015473178,
    6.498253906,2.457162354,0.182386002
  ),
  byrow = TRUE,
  nrow = 25
)

dataset$coords <- abind::abind(dataset$coords, pachycrocuta1, pachycrocuta2, pachycrocuta3, pachycrocuta4, along = 3)
dataset$labels <- c(dataset$labels, "FN3", "FN3", "FN3", "FN3")

# perform procrustes superimposition in form space

Y.gpa <- gpagen(dataset$coords,
                surfaces = c(6:25))
form_coords <- Y.gpa$coords
for(i in 1:length(Y.gpa$Csize)) {
  form_coords[,,i] <- form_coords[,,i] * Y.gpa$Csize[i]
}

# save the procrustes superimposed coordiantes in a format that can be loaded in python

write.table(vector_from_landmarks(form_coords[,,1:823]), "fn3_reference_landmarks.txt", sep  = ",",
            col.names = FALSE, row.names = FALSE)
write.table(vector_from_landmarks(form_coords[,,824:827]), "fn3_pachycrocuta.txt", sep  = ",",
            col.names = FALSE, row.names = FALSE)
write.table(dataset$labels[1:823], "fn3_reference_labels.txt", sep  = ",",
            col.names = FALSE, row.names = FALSE)

#

# PYTHON ANALYSIS HERE ========================================================================

# Analyse output of VAE and MCMC --------------------------------------------------------------

simulated_pachycrocuta1 <- read.table("reconstructed_pachycrocuta1.txt", header = FALSE, sep = ",")
simulated_pachycrocuta2 <- read.table("reconstructed_pachycrocuta2.txt", header = FALSE, sep = ",")
simulated_pachycrocuta3 <- read.table("reconstructed_pachycrocuta3.txt", header = FALSE, sep = ",")
simulated_pachycrocuta4 <- read.table("reconstructed_pachycrocuta4.txt", header = FALSE, sep = ",")
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
  
  sim_pachy1_length <- c(sim_pachy1_length, euclid(simulated_pachycrocuta_tensor[1,,i],
                                                   simulated_pachycrocuta_tensor[2,,i]))
  sim_pachy1_width <- c(sim_pachy1_width, euclid(simulated_pachycrocuta_tensor[3,,i],
                                                 simulated_pachycrocuta_tensor[4,,i]))
  sim_pachy1_depth <- c(sim_pachy1_depth, depth_estimate(simulated_pachycrocuta_tensor[1,,i],
                                                         simulated_pachycrocuta_tensor[2,,i],
                                                         simulated_pachycrocuta_tensor[5,,i]))
  
}

