def cubic_bspline(x,cen,sup):
    """evaluates the cubic bspline with center at 'cen' and support of 'sup' at spatial coordinates x """
    b=sup*cen-2
    if x>=b/sup and x<(1+b)/sup:
        f=1/6*(sup*x-b)**3
    elif x>=(1+b)/sup and x<(2+b)/sup:
        f=-1/2*(sup*x-b)**3 + 2*(sup*x-b)**2 - 2*(sup*x-b) + 2/3
    elif x>=(2+b)/sup and x<(3+b)/sup:
        f=1/2*(sup*x-b)**3 - 4*(sup*x-b)**2 + 10*(sup*x-b) - 22/3
    elif x>=(3+b)/sup and x<(4+b)/sup:
        f=-1/6*(sup*x-b)**3 + 2*(sup*x-b)**2 - 8*(sup*x-b) + 32/3
    else:
        f=0
        
    return f
#
def temporal_basis(t,Nt,Lt): 
   """evaluates laplace basis at time coordinate 't'. Output is 1xNt temporal basis vector"""
   Phit = [1]*Nt
   import math
   for i in range(Nt):
        Phit[i] = Phit[i]*math.sin(math.pi*(i+1)*(t+Lt)/(2*Lt))/math.sqrt(Lt)
        
   return [1]+Phit
#
def spatial_basis(xs,Ns,mn,mx,sup):
    """xs:D dimensional spatial point. Output is 1xNs^D spatial basis vector"""
    import numpy
    
    D = len(xs)
    Ls = mx-mn
    delta = Ls*sup
    
    #efficient basis evaluation for 2D space
    if D==2:    
        for d in range(D):
            cen = numpy.arange(mn[d],mx[d],Ls[d]/Ns)
            dist = numpy.absolute([i-xs[d] for i in cen])
            if d==0:
                idx0 = [i for i,temp in enumerate(dist,0) if temp<delta[d]/2]
                neigh_cen0 = [cen[i] for i in idx0]
            else:
                idx1 = [i for i,temp in enumerate(dist,0) if temp<delta[d]/2]
                neigh_cen1 = [cen[i] for i in idx1]
        import itertools
        ind_spatial_basis = [i*Ns+j for i,j in itertools.product(idx0,idx1)]
        Phis = [cubic_bspline(xs[0],i,4/delta[0])*cubic_bspline(xs[1],j,4/delta[1]) for i,j in itertools.product(neigh_cen0,neigh_cen1)]
        return [Phis, ind_spatial_basis]
    
    #normal basis evaluation otherwise 
    else:
        Phis = [1]*Ns**D
        j=[0]*D
        for i in range(Ns**D):
            for d in range(D):
                if D>1:
                    cen = numpy.arange(mn[d],mx[d],Ls[d]/Ns)
                    Phis[i] = Phis[i]*cubic_bspline(xs[d],cen[j[d]],4/delta[d])
                else:
                    cen = numpy.arange(mn,mx,Ls/Ns)
                    Phis[i] = Phis[i]*cubic_bspline(xs,cen[j[d]],4/delta)
            j[D-1]+=1
            if D>1:
                for k in range(D-2,-1,-1):
                    if j[k+1]==Ns:
                        j[k+1] = 0
                        j[k]+=1
        return Phis 
#
def spatio_temporal_basis(x,mn,mx,Ns,Nt,sup):
    """Evaluates the whole spatio-temporal basis at (D+1) dimensional point x. x(1:D) = spatial_coord, x(end) = time_coord
    along with the constant mean term"""
    from scipy.sparse import csr_matrix
    
    D = len(x) 
    Ds = D-1
    xs = x[0:Ds]
    t = x[D-1]
    Lt = mx[D-1]-mn[D-1]
    import numpy
    
    Phit = temporal_basis(t,Nt,Lt)
    
    #efficient sparse basis evaluation for 2D space + time
    if Ds==2:
        [Phis, ind_spatial_basis] = spatial_basis(xs,Ns,mn[0:Ds],mx[0:Ds],sup)
        ind_full_basis = [0]*len(ind_spatial_basis)*(Nt+1)
        for i in range(len(ind_spatial_basis)):
            for j in range(Nt+1):
                ind_full_basis[i*(Nt+1)+j] = ind_spatial_basis[i]*(Nt+1)+j
        
        Phi = numpy.kron(Phis,Phit)
        
        #appending 1 for the constant mean term i.e. u(s,t)=1
        alpha = numpy.insert(Phi,0,1) 
        
        #adjusting indices
        ind_alpha = [0] + [temp+1 for temp in ind_full_basis]
        row = [0]*len(ind_alpha)
        #creating sparse alpha matrix of size Ns^Ds(Nt+1)+1 x 1 with non-zero entries in alpha at indices ind_alpha
        alpha_sparse = csr_matrix((alpha, (row, ind_alpha)), shape=(1,Ns**Ds*(Nt+1)+1))
        
    
    #normal basis evaluation otherwise     
    else:
        Phis = spatial_basis(xs,Ns,mn[0:Ds],mx[0:Ds],sup)
        Phi = numpy.kron(Phis,Phit)
        #appending 1 for the constant mean term i.e. u(s,t)=1
        alpha = numpy.insert(Phi,0,1)
        ind_alpha = [i for i,j in enumerate(alpha,0) if alpha[i]!=0]
        alpha = [alpha[i] for i in ind_alpha]
        row = [0]*len(ind_alpha)
        alpha_sparse = csr_matrix((alpha, (row, ind_alpha)), shape=(1,Ns**Ds*(Nt+1)+1))
        
    return [alpha_sparse, ind_alpha]

     




        
         
  
