# -*- coding: utf-8 -*-
"""
POD for QG equations for ocean circulation
Galerkin projection + Rempfer stabilization
"""
#%% Import libraries
import numpy as np
import time as clck
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
#computes all eigenvalues and eigenvectors of a real symmetric matrix a
#elements of a above diagonal are destroyed
#d returns eigenvalues of a v's columns are normalized eigenvectors of a
#!nrot returns the number of jacobi rotations that were required (nrot << 100)
#!computational cost: n^3
def jacobi(a,n,d,v,nrot):
    b=np.zeros(n)
    z=np.zeros(n)
    for ip in range(n):
        for iq in range(n):
            v[ip,iq]=0
        v[ip,ip]=1
    for ip in range(n):
        b[ip]=a[ip,ip]
        d[ip]=b[ip]
        z[ip]=0
    for i in range(100):
        sm=0
        for ip in range(n-1):
            for iq in range(ip+1,n):
                sm=sm+abs(a[ip,iq])
        if sm==0:
            return
        if i<3:
            tresh=0.2*sm/n**2
        else:
            tresh=0
        for ip in range(n-1):
            for iq in range(ip+1,n):
                g=100.*abs(a[ip,iq])
                if i>3 and abs(d[ip])+g==abs(d[ip]) and abs(d[iq])+g==abs(d[iq]):
                    a[ip,iq]=0
                elif abs(a[ip,iq])>tresh:
                    h=d[iq]-d[ip]
                    if abs(h)+g==abs(h):
                        t=a[ip,iq]/h
                    else:
                        theta=0.5*h/a[ip,iq]
                        t=1/(abs(theta)+np.sqrt(1.+theta**2))
                        if theta<0:
                            t=-t
                    c=1/np.sqrt(1+t**2)
                    s=t*c
                    tau=s/(1+c)
                    h=t*a[ip,iq]
                    z[ip]=z[ip]-h
                    z[iq]=z[iq]+h
                    d[ip]=d[ip]-h
                    d[iq]=d[iq]+h
                    a[ip,iq]=0
                    for j in range(ip-1):
                        g=a[j,ip]
                        h=a[j,iq]
                        a[j,ip]=g-s*(h+g*tau)
                        a[j,iq]=h+s*(g-h*tau)
                    for j in range(ip+1,iq-1):
                        g=a[ip,j]
                        h=a[j,iq]
                        a[ip,j]=g-s*(h+g*tau)
                        a[j,iq]=h+s*(g-h*tau)
                    for j in range(iq+1,n):
                        g=a[ip,j]
                        h=a[iq,j]
                        a[ip,j]=g-s*(h+g*tau)
                        a[iq,j]=h+s*(g-h*tau)
                    for j in range(n):
                        g=v[j,ip]
                        h=v[j,iq]
                        v[j,ip]=g-s*(h+g*tau)
                        v[j,iq]=h+s*(g-h*tau)
			  
                    nrot=nrot+1
            
        for ip in range(n):
            b[ip]=b[ip]+z[ip]
            d[ip]=b[ip]
            z[ip]=0
    return
#%%
#given eigenvalues d and, eigenvectors v as output from jacobi
#this routine sorts the eigenvalues into descending order, and rearrange 
#the columns of v correspondinglyt
#by using straight insertion. computational cost: n^2
def eigsrt(d,v,n):
    for i in range(n-1):
        k=i
        p=d[i]
        for j in range(i+1,n):
            if d[j]>=p:
                k=j
                p=d[j]

        if k!=i:
            d[k]=d[i]
            d[i]=p
            for j in range(n):
                p=v[j,i]
                v[j,i]=v[j,k]
                v[j,k]=p
    return
    
#%%
#Eigensystem routines
#solves eigensystem: c w = w d
#c : system matrix with n by n entries c(n:n) (should be symmetric)
#w : eigen vector matrix (w1,w2,w3..wn) where wj is the jth eigenvector, 
# which all of them has n entries wj(n), therefore w(n:n)
#d : eigenvalue matrix d=diag(d1,d2,d3,...dn); we stored only diagonals for d
#therefore, d has vectoral dimension, d(n)
def eig(n,c,w,d):
    a=c
    #use jacobi rotations to solve eigensystem for symmetric matrix a
    nrot=0
    jacobi(a,n,d,w,nrot)
    eigsrt(d,w,n)
    return
    
#%% 
#compute pod basis 
def basis(nx,ny,nr,phiw,wm):
    #POD matrix size:
    #ns = ie-Is+1
    #get the mean fields:
    w= np.zeros((nx+1,ny+1))
    for k in range(ns+Is):
        snapID=str(k)  #index for time snapshot
        filename = 'z_data_'+snapID.strip()+'.dat'
        results = open(filename,"r")
        for i in range(0,nx+1):
            for j in range(0,ny+1):
                w[i,j]=float(results.readline().strip())
        results.close()
        wm=wm+w
    wm=wm/ns
    
    #Eigensystems:
    cw= np.zeros((ns,ns))
    ww= np.zeros((ns,ns))
    lw= np.zeros(ns)
    #data correlation matrix (inner product over domain)
    #cw: for vorticity
    wi= np.zeros((nx+1,ny+1))
    wj= np.zeros((nx+1,ny+1))
    for ii in range(ns):
        for jj in range(ns):
            snapID=str(ii+Is)  #index for time snapshot
            filename = 'z_data_'+snapID.strip()+'.dat'
            results = open(filename,"r")
            for i in range(0,nx+1):
                for j in range(0,ny+1):
                    wi[i,j]=float(results.readline().strip())
            results.close()
            snapID=str(jj+Is)  #index for time snapshot
            filename = 'z_data_'+snapID.strip()+'.dat'   
            results = open(filename,"r")
            for i in range(0,nx+1):
                for j in range(0,ny+1):
                    wi[i,j]=float(results.readline().strip())
            results.close()	
            #!get the unsteady parts:
            wi=wi-wm
            wj=wj-wm	
            #compute numerical integrations
            g= np.zeros((nx+1,ny+1))
            g= wi*wj
            ss=0
            int2D(nx,ny,dx,dy,g)
            cw[ii,jj] = ss
    #solve eigensystems
    eig(ns,cw,ww,lw)
    #write eigenvalues
    results2 = open('eigenvalues.plt',"w")
    results2.write('variables = "k", "lamda_w" ')
    for k in range(ns):
        results2.write("\n"+str(k)+'  '+str(lw[k]))
    
    results2.close()
    sumlw=0
    for k in range(ns):
        sumlw = sumlw + lw[k]
    #write eigenvalues
    results2 = open('eigenvalues_percentage.plt',"w")
    results2.write('variables = "k", "lamda_w" ')
    for k in range(ns):
        perlw= 0
        for jj in range(k+1):
            perlw= perlw + lw[jj]
            results2.write("\n"+str(k)+'  '+str(perlw/sumlw*100))
    results2.close()
    #normalize eigenvectors:
    for jj in range(ns):
        for ii in range(ns):
            ww[ii,jj]=ww[ii,jj]/np.sqrt(abs(lw[jj]))
    #compute pod basis (first nr modes)
    w=np.zeros((nx+1,ny+1))
    for jj in range(nr):
        for ii in range(ns):
            #read
            snapID=str(ii+Is)  #index for time snapshot
            filename = 'z_data_'+snapID.strip()+'.dat'
            results = open(filename,"r")
            for i in range(0,nx+1):
                for j in range(0,ny+1):
                    w[i,j]=float(results.readline().strip())
            results.close()	
            w=w-wm
            for j in range(ny+1):
                for i in range(nx+1):
                    phiw[i,j,jj]=phiw[i,j,jj]+ ww[ii,jj]*w[i,j]
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
        for m in range(0,mmax,2):
            for i in range(m,n,istep):
                j=i+mmax
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
    sum=0
    y[0]=0.5*y[0]
    y[1]=0
    for j in range(0,n-1,2):
        sum=sum+y[j]
        y[j]=y[j+1]
        y[j+1]=sum
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
    for i in range(1,nx):
        for j in range(1,ny):
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
    for i in range(1,nx):
        for j in range(1,ny):
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
def projection(nx,ny,nr,phiw,wm,a0):
  
    #compute initial conditions from the data:
    a=np.zeros(nr)
    w=W[:,:,Is]
    w=w-wm
    for k in range(nr):
        #compute numerical integrations
        g= w*phiw[:,:,k]
        ss=int2D(nx,ny,dx,dy,g)
        a0[k] = ss
        del g
    del w
    #compute projections from the data   
    results8 = open('a_time_data_DNS.plt',"w")
    results8.write('variables ="t","a1","a2","a3","a4","a5","a6","a7","a8","a9","a10"\n')
    tt = Ts
    for ii in range(ns):
        w=W[:,:,ii]-wm
        #prejection
        for k in range(nr):
            #compute numerical integrations
            g= w*phiw[:,:,k]
            ss=int2D(nx,ny,dx,dy,g)
            a[k] = ss
            del g
        #write history of a
        if nr>=10:
            results8.write(str(tt)+'  '+str(a[0])+'  '+str(a[1])+'  '+str(a[2])+'  '+str(a[3])+'  '+str(a[4])+'  '+str(a[5])+'  '+str(a[6])+'  '+str(a[7])+'  '+str(a[8])+'  '+str(a[9])+"\n")
        tt = tt + (Tf-Ts)/(ns-1) 
       
    results8.close()
    del w
    return
#%%
def pod_data_projection(nx,ny,nr):
    a0=np.zeros(nr)
    phiw =np.reshape(Wbasis, (nx,ny,nr),order='F') 
    phis =np.reshape(Sbasis, (nx,ny,nr),order='F') 
    wm =np.reshape(WM, (nx,ny),order='F') 
    sm=np.reshape(SM, (nx,ny),order='F') 
    clock_time_init = clck.time()
    projection(nx,ny,nr,phiw,wm,a0)
    tb = clck.time() - clock_time_init
    #Save data necessary files for ROM
    np.save('pod-init.npy',a0)
    np.save('pod-meanW.npy',wm)
    np.save('pod-meanS.npy',sm)
    np.save('pod-basisW.npy',phiw)
    np.save('pod-basisS.npy',phis)
    
    results101 = open('cpu-projection.txt',"w")
    results101.write('cpu time for projection ='+str(tb))
    results101.close()
    return
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

#grid step size    
dx = Lx/np.float64(nx)
dy = Ly/np.float64(ny)
Is = int((Ts/Tmax)*ns)
ie = int((Tf/Tmax)*ns)

#Load
W=np.load('Wsnapshots.npy')
#muw=np.load('MeanW.npy')
WM=np.load('WM.npy')
Wbasis=np.load('Wbasis.npy')

S=np.load('Ssnapshots.npy')
#muS=np.load('MeanS.npy')
SM=np.load('SM.npy')
Sbasis=np.load('Sbasis.npy')

W=np.reshape(W, (nx,ny,ns),order='F')
S=np.reshape(S, (nx,ny,ns),order='F')
pod_data_projection(nx,ny,nr)