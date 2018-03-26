def func_newsample_covlearn( y_n, alpha_sparse_n, ind_alpha_n, Gamma_sparse, rho, kappa, w_hat, n, L, U):
    """Online Learning and resulting weights-Dave Zachariah"""
    #n : training point number n =1,2,...
    #L: the number of iterations per training point
    #U: =1 which implies a constant term for overall mean of space-time process in the model
    
    #Libraries
    from math import sqrt,sin,pi
    from scipy.sparse import csr_matrix
    import cmath
    
    #Recursive update
    Gamma_sparse = Gamma_sparse + alpha_sparse_n.T*alpha_sparse_n
    rho = rho + (alpha_sparse_n.T*y_n).toarray()
    kappa = kappa + abs(y_n)**2
    
    #Common variables
    eta  = kappa + ((w_hat.T*Gamma_sparse).dot(w_hat)) - 2*(w_hat.T.dot(rho))
    eta = eta[0,0]
    zeta = rho - Gamma_sparse*w_hat
    
    #Cycle
    for rep in range(L):
        for i in ind_alpha_n:
            psi   = zeta[i,0] + Gamma_sparse[i,i]*w_hat[i,0]
            if psi >= 0:
                s_hat = 1
            else:
                s_hat = -1
            
            #Compute alpha, beta, gamma
            alpha = eta + Gamma_sparse[i,i]*(w_hat[i,0])**2 + 2*( w_hat[i,0]*zeta[i,0])
            beta  = Gamma_sparse[i,i]
            gamma = abs(psi)
            
            #Update estimate
            w_hat_i_new = 0
            
            if beta> 10**7*sin(pi): #to avoid division by a very small number close to zero 
                if i==0:
                    if beta > 0:
                        w_hat_i_new = psi/beta 
                else: 
                    if (n-1)*gamma**2 > ( alpha*beta - gamma**2 ):
                        if (alpha*beta - gamma**2)/(n-1)<0:
                            r_star = ( (gamma/beta) - (1/beta) * cmath.sqrt( (alpha*beta - gamma**2)/(n-1) ) ).real
                        else:
                            r_star = ( (gamma/beta) - (1/beta) * sqrt( (alpha*beta - gamma**2)/(n-1) ) ).real 
                            #ensure numerically real-valued
                        
                        w_hat_i_new = r_star*s_hat
            
            #Update common varaibles
            eta  = eta + beta * (w_hat[i,0] - w_hat_i_new )**2 + 2*((w_hat[i,0] - w_hat_i_new )*zeta[i,0])
            A = csr_matrix.getrow(Gamma_sparse.T,i)
            zeta = zeta + (A.T*(w_hat[i,0] - w_hat_i_new)).toarray()

            #Store update
            w_hat[i,0] = w_hat_i_new
    
    return[w_hat, Gamma_sparse, rho, kappa]
            
                
            
            
            
            
            
                
        





