# -*- coding: utf-8 -*-

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time as clck
#%%
def tdma(a,b,c,r,x,s,e):
    #forward elimination phase
    for i in range(s+1,e):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]

    #backward substitution phase 
    x[e-1] = r[e-1]/b[e-1]
    for i in range(e-2,s-1,-1):
        x[i] = (r[i]-c[i]*x[i+1])/b[i]
    return
#%%
def c4d(u,up,h,n):  
    i=0
    a=np.zeros(n)
    b=np.zeros(n)
    c=np.zeros(n)
    r=np.zeros(n)
    b[i] = 1
    c[i] = 2
    r[i] = (-5*u[i] + 4*u[i+1] + u[i+2])/(2*h)
    
    for i in range(1,n-1):
        a[i] = 1/4
        b[i] = 1
        c[i] = 1/4
        r[i] = 3/2*(u[i+1]-u[i-1])/(2*h)
   
    i=n-1
    a[i] = 2
    b[i] = 1
    r[i] = (-5*u[i] + 4*u[i-1] + u[i-2])/(-2*h)
    tdma(a,b,c,r,up,1,n)
    return
#%%
def c4dd(u,upp,h,n):
    i=0
    a=np.zeros(n)
    b=np.zeros(n)
    c=np.zeros(n)
    r=np.zeros(n)
    b[i] = 1
    c[i] = 11
    r[i] = (13*u[i]-27*u[i+1]+15*u[i+2]-u[i+3])/(h*h)
    
    for i in range(n-1):
        a[i] = 1/10
        b[i] = 1
        c[i] = 1/10
        r[i] = 6/5*(u[i-1]-2*u[i]+u[i+1])/(h*h) 

    i=n-1
    a[i] = 11
    b[i] = 1
    r[i] = (13*u[i]-27*u[i-1]+15*u[i-2]-u[i-3])/(h*h)
    tdma(a,b,c,r,upp,1,n)
    
    return
#%%
def lin_ope_H(nx,ny,dx,u,hu):
    #ux
    a=np.zeros(nx)
    b=np.zeros(nx)
    for j in range(ny):
        for i in range(nx):
            a[i] = u[i,j]
        c4d(a,b,dx,nx)     
        for i in range(nx):
            hu[i,j] = b[i]
    del a,b
    #recompute if iord=1 (2nd-order scheme)
    if iord==1:
        hu[1:nx-1,0:ny] = (u[2:nx,0:ny]-u[0:nx-2,0:ny])/(2*dx)    
    return
#%%
def non_ope_N(nx,ny,dx,dy,w,s,nu):
    #convective term:
    a=np.zeros((nx,ny))
    e=np.zeros((nx,ny))
    #sy
    a=np.zeros(ny)
    b=np.zeros(ny)
    for i in range(nx):
        for j in range(ny):
            a[j] = s[i,j]
        c4d(a,b,dy,ny)     
        for j in range(ny):
            e[i,j] = b[j]
    del a,b  

    #wx
    a=np.zeros(nx)
    b=np.zeros(nx)
    for j in range(ny):
        for i in range(nx):
            a[i] = w[i,j]
        c4d(a,b,dx,nx)     
        for i in range(nx):
            nu[i,j]=-e[i,j]*b[i]
    del a,b
 
    #sx
    a=np.zeros(nx)
    b=np.zeros(nx)
    for j in range(ny):
        for i in range(nx):
            a[i] = s[i,j]
        c4d(a,b,dx,nx)     
        for i in range(nx):
            e[i,j]=b[i]
    del a,b
    #wy
    a=np.zeros(ny)
    b=np.zeros(ny)
    for i in range(nx):
        for j in range(ny):
            a[j] = w[i,j]
        c4d(a,b,dy,ny)     
        for j in range(ny):
            nu[i,j] = nu[i,j] + e[i,j]*b[j]
    del a,b
    del e

    #recompute if iord=1 (2nd-order Arakawa scheme)
    if iord==1:
        hh = 1/3
        gg = 1/(4*dx*dy)
        for j in range(1,ny-1):
            for i in range(1,nx-1):
                j1 =gg*((w[i+1,j]-w[i-1,j])*(s[i,j+1]-s[i,j-1])-(w[i,j+1]-w[i,j-1])*(s[i+1,j]-s[i-1,j]) )
                j2 =gg*(w[i+1,j]*(s[i+1,j+1]-s[i+1,j-1])-w[i-1,j]*(s[i-1,j+1]-s[i-1,j-1])-w[i,j+1]*(s[i+1,j+1]-s[i-1,j+1])+w[i,j-1]*(s[i+1,j-1]-s[i-1,j-1]))
                j3 =gg*(w[i+1,j+1]*(s[i,j+1]-s[i+1,j])-w[i-1,j-1]*(s[i-1,j]-s[i,j-1])-w[i-1,j+1]*(s[i,j+1]-s[i-1,j])+w[i+1,j-1]*(s[i+1,j]-s[i,j-1]) )
                nu[i,j] = - (j1+j2+j3)*hh  #Jacobian

    #recompute if iord=2 (4th-order Arakawa scheme)
    if iord==2:
        hh = 1/3
        zz = 2/3
        ee = 1/(8*dx*dy)
        gg = 1/(4*dx*dy)
        for j in range(2,ny-2):
            for i in range(2,nx-2):
                j1 =gg*((w[i+1,j]-w[i-1,j])*(s[i,j+1]-s[i,j-1])-(w[i,j+1]-w[i,j-1])*(s[i+1,j]-s[i-1,j]) )
                j2 =gg*(w[i+1,j]*(s[i+1,j+1]-s[i+1,j-1])-w[i-1,j]*(s[i-1,j+1]-s[i-1,j-1])-w[i,j+1]*(s[i+1,j+1]-s[i-1,j+1])+w[i,j-1]*(s[i+1,j-1]-s[i-1,j-1]))
                j3 =gg*(w[i+1,j+1]*(s[i,j+1]-s[i+1,j])-w[i-1,j-1]*(s[i-1,j]-s[i,j-1])-w[i-1,j+1]*(s[i,j+1]-s[i-1,j])+w[i+1,j-1]*(s[i+1,j]-s[i,j-1]) )
                j11=ee*((w[i+1,j+1]-w[i-1,j-1])*(s[i-1,j+1]-s[i+1,j-1])-(w[i-1,j+1]-w[i+1,j-1])*(s[i+1,j+1]-s[i-1,j-1]))
                j22=ee*(w[i+1,j+1]*(s[i,j+2]-s[i+2,j])-w[i-1,j-1]*(s[i-2,j]-s[i,j-2])-w[i-1,j+1]*(s[i,j+2]-s[i-2,j])+w[i+1,j-1]*(s[i+2,j]-s[i,j-2]))
                j33=ee*(w[i+2,j]*(s[i+1,j+1]-s[i+1,j-1])-w[i-2,j]*(s[i-1,j+1]-s[i-1,j-1])-w[i,j+2]*(s[i+1,j+1]-s[i-1,j+1])+w[i,j-2]*(s[i+1,j-1]-s[i-1,j-1]))
                nu[i,j]=-(zz*(j1+j2+j3)-(j11+j22+j33)*hh)    #Jacobian  
    return
#%%
#---------------------------------------------------------------------------!
#Laplacian operator L[u] = dxx_u + dyy_u
#---------------------------------------------------------------------------!
def lin_ope_L(nx,ny,dx,dy,u,lu):
    # viscous terms for vorticity transport equation:
    #uxx
    a=np.zeros(nx)
    b=np.zeros(nx)
    for j in range(ny):
        for i in range(nx):
            a[i] = u[i,j]
        c4dd(a,b,dx,nx)     
        for i in range(nx):
            lu[i,j] = b[i]
    del a,b

    # uyy
    a=np.zeros(ny)
    b=np.zeros(ny)
    for i in range(nx):
        for j in range(ny):
            a[j] = u[i,j]
        c4dd(a,b,dy,ny)     
        for j in range(ny):
            lu[i,j] =lu[i,j]+ b[j]
    del a,b    

    #recompute if iord=1 (2nd-order scheme)
    if iord==1:
        lu[1:nx-1,1:ny-1] = ((u[2:nx,1:ny-1]-2*u[1:nx-1,1:ny-1]+u[0:nx-2,1:ny-1])/(dx*dx)+(u[1:nx-1,2:ny]-2*u[1:nx-1,1:ny-1]+u[1:nx-1,0:ny-2])/(dy*dy))  
    return
#%%
#5th order scheme for numerical integration of g(i,j)
#4th order on the boundaries
#for equally distributed mesh with interval dx and dy dual integration
def hint2D(nx,ny,dx,dy,g):
    sy=np.zeros(ny)
    th=dx/24
    for j in range(ny):
        sy[j] = 0
        for i in range(0,nx-2):
            ds = th*(-g[i-1,j]+13*g[i,j]+13*g[i+1,j]-g[i+2,j])
            sy[j] = sy[j] + ds
        #boundaries
        i=0
        sy[j]=sy[j]+2*th*(5*g[i,j]+8*g[i+1,j]-g[i+2,j])
        i=nx-1
        sy[j]=sy[j]+2*th*(5*g[i+1,j]+8*g[i,j]-g[i-1,j])
    th=dy/24
    s=0
    for j in range(1,ny-2):
        ds=th*(-sy[j-1]+13*sy[j]+13*sy[j+1]-sy[j+2])
        s=s+ds
    #!boundaries
    j=0
    s=s+2*th*(5*sy[j]+8*sy[j+1]-sy[j+2])
    j=ny-1
    s=s+2*th*(5*sy[j+1]+8*sy[j]-sy[j-1])
    return s
#%%
#Simpson's 1/3 rule for numerical integration of g(i,j)
#for equally distributed mesh with interval dx and dy
#n should be power of 2 dual integration
def simp2D(nx,ny,dx,dy,g):
    sy=np.zeros(ny)
    nh = nx//2
    th = 1/3*dx
    for j in range(ny):
        sy[j] = 0
        for i in range(0,nh):
            ds =th*(g[2*i,j]+4*g[2*i+1,j]+g[2*i+2,j])
            sy[j] = sy[j] + ds
    nh=ny//2
    th=1/3*dy
    s=0
    for j in range(nh):
        ds = th*(sy[2*j]+4*sy[2*j+1]+sy[2*j+2])
        s=s+ds
    return s
        
#%%
#trapezoidal rule for numerical integration of g(i,j)
#for equally distributed mesh with interval dx and dy
#dual integration
def trap2D(nx,ny,dx,dy,g):
    sy=np.zeros(ny)
    th = 0.5*dx
    for j in range(ny):
        sy[j] = 0
        for i in range(nx-1):
            ds = th*(g[i,j]+g[i+1,j])
            sy[j] = sy[j] + ds
    th = 0.5*dy
    s = 0
    for j in range(ny-1):
        ds = th*(sy[j]+sy[j+1])
        s = s + ds
    return s
#%% 
#numerical integration of g(i,j)
def int2D(nx,ny,dx,dy,g):
    if irule==1: #trapezoidal
        s=trap2D(nx,ny,dx,dy,g)
    elif irule==2: #simphson
        s=simp2D(nx,ny,dx,dy,g)
    else:       #5th order
        s=hint2D(nx,ny,dx,dy,g)
    return s
#%%
#---------------------------------------------------------------------------!
# build coefficient for Galerkin system 
#---------------------------------------------------------------------------!
def coeff(nx,ny,nr,phiw,phis,wm,sm,c1,c2,c3,c1b,c2b):
    #constant terms:
    lwm=np.zeros((nx,ny))
    hsm=np.zeros((nx,ny))
    num=np.zeros((nx,ny))
    lin_ope_L(nx,ny,dx,dy,wm,lwm)
    lin_ope_H(nx,ny,dx,sm,hsm)
    non_ope_N(nx,ny,dx,dy,wm,sm,num)


    for k in range(nr):
        #Viscous
        g=lwm*phiw[:,:,k]
        ss=int2D(nx,ny,dx,dy,g)
        c1[k] = ss*(1/Re)*(1+ k/nr*va) 
        c1b[k] = ss
        del g  
    	#Ekman
        g=wm*phiw[:,:,k]
        ss=int2D(nx,ny,dx,dy,g)
        c1[k] = c1[k] - ss*St
        del g
        #Coriolis
        g=hsm*phiw[:,:,k]
        ss=int2D(nx,ny,dx,dy,g)
        c1[k] = c1[k] + ss/Ro
        del g

        #Jacobian
        g= num*phiw[:,:,k] 
        ss=int2D(nx,ny,dx,dy,g)
        c1[k] = c1[k] + ss
        del g
      
	    #Forcing:
        g= (np.sin(np.pi*(-1+np.arange(ny)*dy))/Ro)*phiw[:,:,k]
        ss=int2D(nx,ny,dx,dy,g)
        c1[k] = c1[k] + ss
        del g

    #linear terms: 
    lq=np.zeros((nx,ny))
    lp=np.zeros((nx,ny))
    numq=np.zeros((nx,ny))
    nqum=np.zeros((nx,ny))
    qi=np.zeros((nx,ny))
    qj=np.zeros((nx,ny))
    qij=np.zeros((nx,ny))
    for i in range(nr):
        q=phiw[:,:,i]
        p=phis[:,:,i]
        lin_ope_L(nx,ny,dx,dy,q,lq)
        lin_ope_H(nx,ny,dx,p,lp)
        non_ope_N(nx,ny,dx,dy,q,sm,numq)
        non_ope_N(nx,ny,dx,dy,wm,p,nqum)
        #inner products
        for k in range(nr):
            #Viscous
            g= lq*phiw[:,:,k]
            ss=int2D(nx,ny,dx,dy,g)
            c2[i,k] = ss*(1/Re)*(1+k/nr*va)
            c2b[i,k] = ss
       
    		#Ekman
            g=q*phiw[:,:,k]
            ss=int2D(nx,ny,dx,dy,g)
            c2[i,k]=c2[i,k]- ss*St    
            #Coriolis
            g= lp*phiw[:,:,k]
            ss=int2D(nx,ny,dx,dy,g)
            c2[i,k]=c2[i,k] + ss/Ro
            #Jacobian
            g= (numq+nqum)*phiw[:,:,k]
            ss=int2D(nx,ny,dx,dy,g)
            c2[i,k]=c2[i,k]+ss



    #nonlinear terms:
    for i in range(nr):
        qi= phiw[:,:,i]
        for j in range(nr):
            qj = phis[:,:,j]
            non_ope_N(nx,ny,dx,dy,qi,qj,qij)
            for k in range(nr):
                #inner product
                g=qij*phiw[:,:,k]
                ss=int2D(nx,ny,dx,dy,g)
                c3[i,j,k] = ss
                del g
    del lwm,hsm,num,q,lq,p,lp,numq,nqum,qi,qj,qij
    return
#%%
#FFT routine for 1-dimensional data 
def four1(data,nn,isign):
    n=2*(nn-1)
    j=0
    for i in range(0,n,2):
        if j>i:
            tempr=data[j]
            tempi=data[j+1]
            data[j]=data[i]
            data[j+1]=data[i+1]
            data[i]=tempr
            data[i+1]=tempi
        m=n//2-1
        while ((m>=1) and (j>m)):
            j=j-m
            m=m//2
        j=j+m
    mmax=1
    while n>mmax:
        istep=2*mmax
        theta=6.28318530717959/(isign*mmax)
        wpr=-2*np.sin(0.5*theta)**2
        wpi=np.sin(theta)
        wr=1
        wi=0
        for m in range(0,mmax-2,2):
            for i in range(m,n-2,istep):
                j=i+mmax
                if (j+1)<data.shape[0]:
                    tempr=wr*data[j]-wi*data[j+1]
                    tempi=wr*data[j+1]+wi*data[j]
                    data[j]=data[i]-tempr
                    data[j+1]=data[i+1]-tempi
                    data[i]=data[i]+tempr
                    data[i+1]=data[i+1]+tempi
            wtemp=wr
            wr=wr*wpr-wi*wpi+wr
            wi=wi*wpr+wtemp*wpi+wi
        mmax=istep
    return          
#%%
#computes real fft
def realft(data,n,isign):
    theta=6.28318530717959/2/n
    c1=0.5
    if isign==1:
        c2=-0.5
        four1(data,n,+1)
    else:
        c2=0.5
        theta=-theta
    wpr=-2*np.sin(0.5*theta)**2
    wpi=np.sin(theta)
    wr=1+wpr
    wi=wpi
    n2p3=2*(n-1)+2
    for i in range(1,n//2+1):
        i1=2*i-1
        i2=i1+1
        i3=n2p3-i2
        i4=i3+1
        wrs=wr
        wis=wi
        h1r=c1*(data[i1]+data[i3])
        h1i=c1*(data[i2]-data[i4])
        h2r=-c2*(data[i2]+data[i4])
        h2i=c2*(data[i1]-data[i3])
        data[i1]=h1r+wrs*h2r-wis*h2i
        data[i2]=h1i+wrs*h2i+wis*h2r
        data[i3]=h1r-wrs*h2r+wis*h2i
        data[i4]=-h1i+wrs*h2i+wis*h2r
        wtemp=wr
        wr=wr*wpr-wi*wpi+wr
        wi=wi*wpr+wtemp*wpi+wi
    if isign==1:
        h1r=data[0]
        data[0]=h1r+data[1]
        data[1]=h1r-data[1]
    else:
        h1r=data[0]
        data[0]=c1*(h1r+data[1])
        data[1]=c1*(h1r-data[1])
        four1(data,n,-1)
    return
#%%
#calculates sine transform of a set of n real valued data points, y(1,2,..n)
#y(1) is zero, y(n) not need to be zero, but y(n+1)=0
#also calculates inverse transform, but output should be multiplied by 2/n
#n should be powers of 2
#use four1 and realft routines
def sinft(y,n):
    theta=3.14159265358979/n
    wr=1
    wi=0
    wpr=-2*np.sin(0.5*theta)**2
    wpi=np.sin(theta)
    y[0]=0
    m=n//2
    for j in range(m):
        wtemp=wr
        wr=wr*wpr-wi*wpi+wr
        wi=wi*wpr+wtemp*wpi+wi
        y1=wi*(y[j]+y[n-j-1])
        y2=0.5*(y[j]-y[n-j-1])
        y[j]=y1+y2
        y[n-j-1]=y1-y2
    realft(y,m,+1)
    summ=0
    y[0]=0.5*y[0]
    y[1]=0
    for j in range(0,n-1,2):
        summ=summ+y[j]
        y[j]=y[j+1]
        y[j+1]=summ
    return
#%%
#Compute fast fourier sine transform for 2D data
#Homogeneous Drichlet Boundary Conditios (zero all boundaries)
#Input:: u(0:nx,0:ny) 
#        where indices 0,nx,ny represent boundary data and should be zero
#Output::override
#Automatically normalized
#isign=-1 is inverse transform and 2/N is already applied 
#        (from grid data to fourier coefficient)
#isign=+1 is forward transform
#        (from fourier coefficient to grid data)
def sinft2d(nx,ny,isign,u):
    if isign==-1:   #inverse transform
        v=np.zeros(nx)
        for j in range(ny-1):
            for i in range(nx):
                v[i] = u[i,j]
            sinft(v,nx)
            for i in range(1,nx):
                u[i,j]=v[i]*2/nx
        v=np.zeros(ny)
#compute inverse sine transform to find fourier coefficients of f in y-direction
        for i in range(nx-1):
            for j in range(ny):
                v[j] = u[i,j]
            sinft(v,ny)
            for j in range(1,ny):
                u[i,j]=v[j]*2/ny
    else: #forward transform
        v=np.zeros(nx)
        for j in range(ny-1):
            for i in range(nx):
                v[i] = u[i,j]
            sinft(v,nx)
            for i in range(1,nx):
                u[i,j]=v[i]
        v=np.zeros(ny)

        for i in range(nx-1):
            for j in range(ny):
                v[j] = u[i,j]
            sinft(v,ny)
            for j in range(1,ny):
                u[i,j]=v[j]
    return
#%%
#---------------------------------------------------------------------------!
#Routines for fast Poisson solver
#fast sin transformation direct poisson solver
#fast direct poisson solver for homogeneous drichlet boundary conditions
#using discreate fast sin transformation along x and y axis 
#second order formula
def fst2(nx,ny,dx,dy,f,u):
    ft=f
    #fast inverse fourier sine transform of source term:
    isign=-1
    sinft2d(nx,ny,isign,ft)
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            alpha=2/(dx*dx)*(np.cos(np.pi*i/nx)-1) \
                +2/(dy*dy)*(np.cos(np.pi*j/ny)-1)
            u[i,j]=ft[i,j]/alpha
    isign=1
    sinft2d(nx,ny,isign,u)
    return
#%%
def fst4(nx,ny,dx,dy,f,u):
    beta = dx/dy
    a =-10*(1+beta*beta)
    b = 5- beta*beta
    c=5*beta*beta-1
    d = 0.5*(1+beta*beta)
    ft=f
    isign=-1
    #fast inverse fourier sine transform of source term:
    sinft2d(nx,ny,isign,ft)
    #Compute fourier coefficient of u:
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            alpha= a + 2*b*np.cos(np.pi*i/nx) \
                +2*c*np.cos(np.pi*j/ny) \
                +4*d*np.cos(np.pi*i/nx)*np.cos(np.pi*j/ny)
            gamma=8+2*np.cos(np.pi*i/nx) \
                +2*np.cos(np.pi*j/ny)
            u[i,j]=ft[i,j]*(dx*dx)*0.5*gamma/alpha
    isign=1
    sinft2d(nx,ny,isign,u)
    return
#%%
def construction(nx,ny,nr,a,wm,phiw,w,s):
    #construct vorticity and stream function
    for j in range(ny):
        for i in range(nx):
            w[i,j]=0.0
            for k in range(nr):
                w[i,j] = w[i,j] + a[k]*phiw[i,j,k]
            w[i,j] = wm[i,j]+ w[i,j]

    #compute stream function:
    for j in range(ny):
        for i in range(nx):
            s[i,j]=0.0

    if iord==1:
        fst2(nx,ny,dx,dy,-w,s)
    else:
        fst4(nx,ny,dx,dy,-w,s)
    return
#%%
# def outfield(nx,ny,tt,x,y,s,w):
#     results400.write('zone f=point i='+str(nx)+',j='+str(ny)+',t="time'+str(tt)+'"\n')    
#     for j in range(ny):
#         for i in range(nx):
#             results400.write(str(x[i])+"   "+str(y[j])+"   "+str(s[i,j])+"   "+str(w[i,j])+"\n")
#     return
#%%
def rhs(nr,c1,c2,c3,c1b,c2b,a,r):
    r2=np.zeros(nr)
    r3=np.zeros(nr)
    if imode==1: #dynamic model (compute eddy viscosity dynamically) 
        nrc = nr - deltaR #coarser number of modes
        #compute H(k)
        hh=np.zeros(nr)
        
        for k in range(nr):
            hh[k] = 0.0
            hh[k] = hh[k] + c2[nrc+1:nr,k]*a[nrc+1:nr]  
            hh[k] = hh[k] + c3[0:nr,0:nr,k]*a[0:nr]*a[0:nr]
            hh[k] = hh[k]- c3[0:nrc,0:nrc,k]*a[0:nrc]*a[0:nrc]
        #compute M(k)
        mm=np.zeros(nr)
        mm[0:nr] = mm[0:nr] + c2b[nrc+1:nr,0:nr]*a[nrc+1:nr]  


        #Least-squares
        aa=0.0
        bb=0.0
        for k in range(nr):
            aa=aa-hh[k]*mm[k]
            bb=bb+mm[k]*mm[k]
        #clip
        kappa = max(0,aa/(bb+1e-15))
        c1n = c1 + c1b*kappa
        c2n=c2+c2b*kappa
    else:
        c1n = c1 
        c2n=c2 

    #Compute RHS
    for k in range(nr):
        r2[k] = 0
        for i in range(nr):
            r2[k] = r2[k] + c2n[i,k]*a[i]
    for k in range(nr):
        r3[k] = 0
        for j in range(nr):
            for i in range(nr):
                r3[k] = r3[k] + c3[i,j,k]*a[i]*a[j]
    r= c1n + r2 + r3
    return
#%%
def rk3(nr,dt,a,c1,c2,c3,c1b,c2b):
    a1=np.zeros(nr)
    r=np.zeros(nr)
    rhs(nr,c1,c2,c3,c1b,c2b,a,r)
    a1 = a + dt*r
    rhs(nr,c1,c2,c3,c1b,c2b,a1,r)
    a1=(3/4)*a+ (1/4)*a1+ (1/4)*dt*r
    rhs(nr,c1,c2,c3,c1b,c2b,a1,r)
    a= (1/3)*a+ (2/3)*a1+ (2/3)*dt*r
    del a1,r
    return
#%% Functions
def pod_rom(nx,ny,nr):
    x=np.zeros(nx)
    y=np.zeros(ny)
    a0=np.zeros(nr)
    a=np.zeros(nr)
    aa=np.zeros(nr)
    w=np.zeros((nx,ny))
    s=np.zeros((nx,ny))
    wm=np.zeros((nx,ny))
    sm=np.zeros((nx,ny))
    wa=np.zeros((nx,ny))
    sa=np.zeros((nx,ny))
    phiw=np.zeros((nx,ny,nr))
    phis=np.zeros((nx,ny,nr))
    c1=np.zeros(nr)
    c2=np.zeros((nr,nr))
    c3=np.zeros((nr,nr,nr))
    c1b=np.zeros(nr)
    c2b=np.zeros((nr,nr))
    for i in range(0,nx):
        x[i]=(i+1)*dx
    for j in range(0,ny):
        y[j]=-1+(j+1)*dy

    #read basis and mean values
    a0=np.load('pod-init.npy')
    wm=np.load('pod-meanW.npy')
    sm=np.load('pod-meanS.npy')
    phiw=np.load('pod-basisW.npy')
    phis=np.load('pod-basisS.npy')
    
    clock_time_init = clck.time()
    #compute the coefficients for reduced order model:
    coeff(nx,ny,nr,phiw,phis,wm,sm,c1,c2,c3,c1b,c2b)
    tc = clck.time() - clock_time_init
    
    results8 = open('a_time_data_ROM.plt',"w")  
    results8.write('variables ="t","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10"\n')
    
    results9 = open('a_time_data_ROM_isnap.plt',"w")  
    results9.write('variables ="t","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10"\n')
    
    results400 = open('field-ROM.plt',"w")  
    results400.write('variables ="x","y","s","w"\n')
    
    #solve reduced system:
    #initial conditions
    tt=Ts
    a=a0
    if nr>=10:
        results8.write(str(tt)+'  '+str(a[0])+'  '+str(a[1])+'  '+str(a[2])+'  '+str(a[3])+'  '+str(a[4])+'  '+str(a[5])+'  '+str(a[6])+'  '+str(a[7])+'  '+str(a[8])+'  '+str(a[9])+'  '+"\n")
        results9.write(str(tt)+'  '+str(a[0])+'  '+str(a[1])+'  '+str(a[2])+'  '+str(a[3])+'  '+str(a[4])+'  '+str(a[5])+'  '+str(a[6])+'  '+str(a[7])+'  '+str(a[8])+'  '+str(a[9])+'  '+"\n")

         
    #construct field from reduced system coefficients
    construction(nx,ny,nr,a,wm,phiw,w,s)
    #outfield(nx,ny,tt,x,y,s,w)
    results400.write('zone f=point i='+str(nx)+',j='+str(ny)+',t="time'+str(tt)+'"\n')    
    for j in range(ny):
        for i in range(nx):
            results400.write(str(x[i])+"   "+str(y[j])+"   "+str(s[i,j])+"   "+str(w[i,j])+"\n")
    
    #average a
    tr=0
    #Time integration:
    for k in range(i1+1,i2+1):
        tt = tt+dt
        print(tt,a[0])
        #solver
        t1= clck.time()
        rk3(nr,dt,a,c1,c2,c3,c1b,c2b)
        t2= clck.time()
        tr = tr + t2-t1
    	
        #to compute mean
        aa=aa+a
        
        if k%fhist==0 and nr>=10:
            results8.write(str(tt)+'  '+str(a[0])+'  '+str(a[1])+'  '+str(a[2])+'  '+str(a[3])+'  '+str(a[4])+'  '+str(a[5])+'  '+str(a[6])+'  '+str(a[7])+'  '+str(a[8])+'  '+str(a[9])+"\n")
        if k%fsnap==0:
            results9.write(str(tt)+'  '+str(a[0])+'  '+str(a[1])+'  '+str(a[2])+'  '+str(a[3])+'  '+str(a[4])+'  '+str(a[5])+'  '+str(a[6])+'  '+str(a[7])+'  '+str(a[8])+'  '+str(a[9])+"\n")
        #output
        if k%ffile==0:
            construction(nx,ny,nr,a,wm,phiw,w,s)
            #outfield(nx,ny,tt,x,y,s,w)
            results400.write('zone f=point i='+str(nx)+',j='+str(ny)+',t="time'+str(tt)+'"\n')    
            for j in range(ny):
                for i in range(nx):
                    results400.write(str(x[i])+"   "+str(y[j])+"   "+str(s[i,j])+"   "+str(w[i,j])+"\n")
    
    
    results400.close()
    results8.close()
    results9.close()
      
    #mean field by pod
    aa=aa/(i2-i1+1)
    construction(nx,ny,nr,aa,wm,phiw,wa,sa)
    
    results200 = open('mean-ROM.plt',"w") 
    results200.write('variables ="x","y","s","w"\n')
    results200.write('zone f=point i='+str(nx)+',j='+str(ny)+"\n")
    
    for j in range(ny):
        for i in range(nx):
            results200.write(str(x[i])+'  '+str(y[j])+'  '+str(sa[i,j])+'  '+str(wa[i,j]))
    results200.close()
    
    #final field
    results100 = open('final-ROM.plt',"w") 
    results100.write('variables ="x","y","s","w"\n')
    results100.write('zone f=point i='+str(nx)+',j='+str(ny)+"\n")
    for j in range(ny):
        for i in range(nx):
            results100.write(str(x[i])+'  '+str(y[j])+'  '+str(s[i,j])+'  '+str(w[i,j]))
    results100.close()
    
    results101 = open('cpu-ROM.txt',"w")
    results101.write('cpu time for coefficients          ='+str(tc))
    results101.write('cpu time for rom                   ='+str(tr))
    results101.close()
#%% Main program:
# Inputs
nx=90     #number of points in x
ny=500	   #number of points in y
Lx=1 		#domain size in x
Ly=2 		#domain size in y
dt = 1e-1      #time step
ns=90		# number of snaphshots for 0-Tmax
ni=10		#ni: number of files for 0-Tmax (for video)
nr=10 		#nr: number of modes to use (for ROM)
Tmax=20 	#Tmax; for simulation
Ts=1		#Ts; start time for POD matrix
Tf=10		#Tf; final time for POD matrix
Re=45		#Re (Reynolds number)
Ro=0.0036  	#Ro (Rossby number)
St=0		#St (Stommel number)
va=0		#va: stabilization parameter (Rempfer) (0-10)
deltaR=5   #filter ratio
fhist=10	#time series writing history
iord=1		# [1]2nd-order,[2]4th-order,[3]Compact4
irule=1		#[1]trap,[2]Simp,[3]5th
imode=0		#[0]Rempfer/Galerkin,[1]Dynamic
isc=1		#screen logo
icheck=19   #icheck

dx = Lx/nx
dy = Ly/ny

Is = int((Ts/Tmax)*ns)
ie = int((Tf/Tmax)*ns)

i1 = int(Ts/dt)
i2 = int(Tf/dt)
ffile= int((Tmax/dt)/ni)
nt=int(Tmax/dt)
fsnap= int(nt/ns)
pod_rom(nx,ny,nr)
