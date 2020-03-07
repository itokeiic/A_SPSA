import numpy
def spsa(x0,func,bounds=None,alpha=0.602,gamma=0.101,deltax_0=0.1,a=None,a_min=1.0e-6,c=1.0e-6,stepredf=0.5,gtol=1.0e-5,graditer=1,memsize=100,IniNfeval=0,maxiter=5000,adaptive_step=True,relaxation=True,dynamic_system=False,*args):
    #INPUT
    #x0: starting input vector (as python list), if dynamic_system=True, append 0 in the list
    #func: python function
    #bounds: [(lower_bound_1, upper_bound_1),(lower_bound_2, upper_bound_2),...,(lower_bound_n, upper_bound_n)], n = # of dim. If None, it is automatically set to [(-10,10)]*n
    #alpha: exponential controlling the reduction of step size
    #gamma: exponential controlling the finite differece gradient perturbation magnitude
    #deltax_0: desiried minimum initial perturbation of x0
    #stepredf: factor controlling the recduction of stepsize along stochastic gradient descent if no improvement in objective function was observed in the generated perturbations to compute stochastic gradient
    #gtol: threshold value below which gradient is considered to be zero and therefore converged
    #graditer: number of times gradients are computed to obtain an averaged stochastic gradient
    #IniNfeval: parameter for accounting for number of function evaluation done before reaching this optimization function.  This optimization function will perform total of maxNfeval - IniNfeval function evaluations.
    #maxiter: total number of iteration to be performed.  The optimization terminates when maxNfeval or maxiter is reached
    #adaptive_step: Initial stepsize is automatically reduced if set to True to provide reliable objective value descent
    redcounter=0
    if dynamic_system == False:
        Npar=len(x0)
    else:
        Npar=len(x0)-1
        
    def g_sa(x,func,ck,niter,*args):#stochastic gradient calculation
        p=len(x)
        gsum=0.0
        for m in range(niter):
            delta=scipy.add(2*scipy.floor(scipy.random.uniform(0,2,p)),-1)
                
            # print "delta = ",delta
            xp=x+ck*delta
            xm=x-ck*delta
            if dynamic_system == True:
                xp[-1]=xm[-1]=x[-1]
            yp=func(xp,*args)
            ym=func(xm,*args)
            gsum=gsum+(yp-ym)/(2*ck*delta)
        ghat=gsum/niter;# print 'ghat = ',ghat
        if dynamic_system == True:
            ghat[-1]=0
        return (ghat,yp,ym,xp,xm,delta)
    
    Xmax=list()
    Xmin=list()
    if bounds is None:
        bounds = [(-10.0,10.0)] * Npar
        print 'No bounds specified. Default:(-10,10).'
    if len(bounds) != Npar:
        raise ValueError('Number of parameters Npar != length of bounds')
    for m in range(0,Npar):
        Xmin.append(bounds[m][0])
        Xmax.append(bounds[m][1])

    Nfeval=IniNfeval
    x0=numpy.array(x0)
    history=[]
    historyx=[]
    p=len(x0)
    A=int(0.1*maxiter)
    y0=func(x0,*args); Nfeval=Nfeval+1
    mem=numpy.ones(memsize)*y0
    x=x0.copy()
    print 'initial objective value = ',y0
    x_best=x0.copy();y_best=y0; #y_ave=y0; y_max=y0
    for k in range(0,maxiter):
        if dynamic_system == True:
            x[-1]=k
        ck=c/(k+1)**gamma
        ghat,yp,ym,xp,xm,delta=g_sa(x,func,ck,graditer,*args);Nfeval=Nfeval+graditer*2
        if (k==0):
            if a == None:
                a=deltax_0*(A+1)**alpha/(min(abs(ghat[:Npar])))
            a_ini=a
            print 'ghat0 = ',ghat[:]        
        ak=a/(k+1+A)**alpha
        #y_ave_old=y_ave
        #y_ave=(k+1)*y_ave/(k+2)+max(yp,ym)/(k+2)

        #delta=scipy.add(2*scipy.floor(scipy.random.uniform(0,2,p)),-1)
        #xp=x+ck*delta
        #xm=x-ck*delta
        #yp=func(xp,*args); Nfeval=Nfeval+1
        #print 'yp = ',yp
        #ym=func(xm,*args); Nfeval=Nfeval+1
        #print 'ym = ',ym
        #ghat=(yp-ym)/(2*ck*delta); #print 'ghat = ',ghat
        print 'k: %d, ym = %f, yp = %f, a = %f'%(k,ym,yp,a)
        xold=x.copy()
        x=x-ak*ghat
        for m in range(0,Npar):
            if x[m]<Xmin[m]:
                x[m]=Xmin[m]
            elif x[m]>Xmax[m]:
                x[m]=Xmax[m]
        y=func(x,*args); history.append(list([Nfeval,y])); historyx.append(list(x))#to keep track of convergence history
        mem=numpy.append(mem,numpy.min([ym,yp]))
        mem=numpy.delete(mem,0)
        #if sqrt(scipy.inner(ghat,ghat)) < gtol:
            #print 'converged!'
            #break
        if ym<y_best:
            x_best=xm; #print 'x_best = ',xm
            y_best=ym
            #a=a/stepredf
        if yp<y_best:
            x_best=xp; #print 'x_best = ',xp
            y_best=yp
            #a=a/stepredf
        if adaptive_step == True:
            #if ((yp-y0)>abs(y0)) or ((ym-y0)>abs(y0)):
            if ((y0-min(yp,ym))<0):
            #if (y0-y_ave)<0:
            #if (mem.mean()>y0+c) and (numpy.mod(k,memsize)==0):
                print 'divergence detected. reinitializing.'
                redcounter+=1
                #x=x0.copy()
                x=x_best.copy()
                a=stepredf*a
                if (redcounter > int(0.05*maxiter)) and relaxation:
                #if (a < a_min) and relaxation:
                    print "Too many divergence. Resetting a and relaxing threshold!"
                    a=a_ini
                    #dim=numpy.random.randint(0,Npar)
                    #x[dim]=numpy.random.uniform(bounds[dim][0],bounds[dim][1])
                    y0=min(yp,ym)
                    redcounter=0
                #y_max=max(yp,ym)
                #a=numpy.max([stepredf*a,c])
                #a=(stepredf+c*numpy.random.randn())*a
                 
            # if y_ave < y0:
                # a=a_ini
    y=func(x,*args); Nfeval=Nfeval+1
    history.append(list([Nfeval,y]))
    historyx.append(list(x))
    print 'number of function evaluation: ',Nfeval
    return (x,y,history,historyx,Nfeval)
