## Import Libraries

import basis
import Learning_from_streaming_data as online
from latlon_conversion import latlon_to_xy

import csv
import numpy
from scipy.sparse import csr_matrix


##Initialize by defining the following

#Ds = Nos. of spatial dimensions (e.g. 2 for 2D space)
#Ns = Nos of spatial basis per dimension
#Nt = Nos. of temporal basis per dimension
#mn = 1x(Ds+1) vector containing the minmum value of each dimension
#mx = 1x(Ds+1) vector containing the maximum value of each dimension
#e.g. for 2D space time data -100 to 100 spatial units in each spatial dimension and from t=0 month to t=59 month
#mn = [-100,-100,0], mx = [100,100,59]
#L = mx-mn : range of each dimension
#sup = support of b-spline basis used for space in fraction of L (e.g 0.5 implies a support of 0.5*range of each spatial dimension)
#P = Ns^Ds*(Nt+1)+1
#L: number of iterations of the algorithm per data point (e.g = 5)
U=1 #corresponding to constant term in the model for overall mean of the spatio-temporal process 

##Training data

#train.csv: file containing training data. Each row corresponds to one space_time point SPT_n
#Format of SPT_n for 2D space and time = [y_n, x_n ] = [y_n, s_1, s_2, t] 
#y_n: spatio_temporal qaunity of interest
#s_1: spatial_coordinate of first spatial axis
#s_2: spatial coordinate of second spatial axis
#t: time coordinate

##Initialize parameters for training with zeros

w_hat = numpy.zeros((P,1))
Gamma_sparse = csr_matrix((P,P)) #sparse matrix for memory efficiency
rho = numpy.zeros((P,1))
kappa = 0

##Read train.csv line by line (assuming 2D space) and learn the weights 'w_hat'
n=0 #first point
with open('train.csv') as training_data:
    csvreader = csv.reader(training_data)
    for SPT_n in csvreader:
        y_n = float(SPT_n[0])
        
        #convert longitude/latitude to x-y coordinates, here R is the radius of earth
        [s1,s2]=latlon_to_xy(float(SPT_n[1]),float(SPT_n[2]),R)
        
        t= float(SPT_n[3])
        x_n= [s1,s2,t]
        
        #Evaluate the spatio-temporal basis
        [alpha_sparse_n,ind_alpha_n] = basis.spatio_temporal_basis(x_n,mn,mx,Ns,Nt,sup)
        
        #Update the predictor for this spatio-temporal point
        [w_hat, Gamma_sparse, rho, kappa] = online.func_newsample_covlearn(y_n, alpha_sparse_n, ind_alpha_n, Gamma_sparse, rho, kappa, w_hat, n+1, L, U)
        
        n+=1 #increment by 1 for next point

##data.csv: file containing space_time coordinates to predict at or test data (N number of points)
# SPT_n format : [y_n, x_n] = [y_n, s_1, s_2, t] for test data or [x_n] = [s_1,s_2,t] for real data
y_pred = numpy.zeros((N,1))
n=0
with open('data.csv') as data:
    csvreader = csv.reader(data)
    for SPT_n in csvreader:      
        y_n = float(SPT_n[0])
        
        #convert longitude/latitude to x-y coordinates, here R is the radius of earth
        [s1,s2]=latlon_to_xy(float(SPT_n[1]),float(SPT_n[2]),R)
        
        t= float(SPT_n[3])
        x_n= [s1,s2,t]
        
        #Evaluate the spatio-temporal basis
        [alpha_sparse_n,ind_alpha_n] = basis.spatio_temporal_basis(x_n,mn,mx,Ns,Nt,sup)
        
        y_pred[n]=alpha_sparse_n*w_hat
        
        n+=1



        

        
