# Localized-Spatio-Temporal-Models

The method allows for making prediction of spatio-temporal quantities such as precipitation, air pollution etc. from training data at spatio-temporal coordinates where the quantity of interest in unknown. It is specifically suitable when the tranining data is coming in form of small batches, point by point in a sequential or streaming manner. It is also equally applicable if the entire training data is available. The method makes use of spatio-temporal basis (composed of local b-spline basis in space and Fourier like basis in time) and uses an iterative algorithm for learning the predictor in a sequenctial manner as data comes in. For details see and please cite: Muhammad Osama, Dave Zachariah, Thomas B. Schön, "Learning Localized Spatio-Temporal Models From Streaming Data" http://arxiv.org/abs/1802.03334. 

There are two main files implemented in Python: 'basis.py' and 'Learning_from_streaming_data.py'. An 'example.py' file describes what variables to initialize and how to use these former two files for training and then evaluating the predictor at desired spatio-temporal coordinates. Below we provide a short description of each file.

## Example plots
The uploaded files contain some example plots from our results on real precipitation data. The precipitation_conotur_plot is the contour plot of predicted precipitation over a spatial region for a specific month. The red dots denote the training points. The precipitation_time_series is a comparison of the actual and predicted percipitation over time for spatial point marked by red cross in the contour plot. The black dashed box and dashed line in these plots constitute a contiguous spatio-temporal test region. 

![Contour plot](muhammad-osama.github.com/repository/precipitation_contour_plot.png)

## A short description of variables in each .py file
### Learning_from_streaming_data.py

This function is used to learn the weights 'w_hat' from training data which are used later for evaluating the predictor.

The function is called in a for loop for each training point in the following way: 

[w_hat, Gamma_sparse, rho, kappa] = func_newsample_covlearn(y_n, alpha_sparse_n, ind_alpha_n, Gamma_sparse, rho, kappa, w_hat, n+1, L, U)

y_n: the spatio-temporal quantity of interest

alphas_sparse_n, ind_alpha_n: from basis.py

Gamma_sparse, rho, kappa, w_hat: Have to be initialized with zeros outside the for loop before calling the above function. See example.py file

n+1: see example.py file

L: The number of itereations of the algortihm for each training point

It is assumed that the training data is organized in a csv file in a way described in the example.py file. For example for 2D space time, the nth line in the training.csv represents a traning point with format [y_n, s1_n, s2_n, t_n]:

y_n : spatio-temporal quantity e.g. precipitation at the nth training point

U: mean parameter set to 1 corresponding to constant spatio-temporal mean

### basis.py
The basis.py evaluates a spatio-temporal basis at a specific spatio-temporal point. The function is called in the following way:

[alpha_sparse_n,ind_alpha_n] = basis.spatio_temporal_basis(x_n,mn,mx,Ns,Nt,sup)

x_n: Dx1 array containing spatio-temporal coordinates e.g. for 2D space time: x_n = [s1, s2, t] where [s1,s2] are the spatial coordinates in Euclidean space and 't' is the time coordinates. If spatial coordinates are [longitude, latitude] they must be converted to Euclidean space. We provide a function latlon_conversion.py to do that

mn: Dx1 array containing the lower limit of each dimension

mx: Dx1 array containing the upper limit of each dimension

Ns: the number of basis in space per dimension

Nt: the number of basis in time

sup: the support of local b-spline spatial basis in fraction of range of dimension

alpha_sparse_n: Ns^(D-1)*(Nt^2+1)x1 sparse array of spatio-temporal basis for the nth point

ind_alpha_n: array containing indices of the non-zero elements in alpha_sparse_n




