import os
import sys
import scipy.io as sio

from readfiles import *

EPS = np.finfo(float).eps

whiskers=[
    "RA0","RA1","RA2","RA3","RA4",
    "RB0","RB1","RB2","RB3","RB4",#"RB5",
    "RC0","RC1", "RC2","RC3","RC4","RC5",
    "RD0","RD1","RD2","RD3","RD4","RD5",
    "RE1","RE2","RE3","RE4","RE5"]

whisker_id=[(0,0),(0,1),(0,2),(0,3),(0,4),
    (1,0),(1,1),(1,2),(1,3),(1,4),#(1,5),
    (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),
    (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),
    (4,1),(4,2),(4,3),(4,4),(4,5)]

def smooth(x,window_len=11,window='hanning'):
    # https://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth.html

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]


def get_xyzc(dirname, whiskername):

    filename = dirname + '/kinematics/x/' + whiskername + '.csv'
    x = read_csv(filename)

    filename = dirname + '/kinematics/y/' + whiskername + '.csv'
    y = read_csv(filename)

    filename = dirname + '/kinematics/z/' + whiskername + '.csv'
    z = read_csv(filename)

    filename = dirname + '/kinematics/c/' + whiskername + '.csv'
    c = read_csv(filename)
    x =  np.array(x).T
    y =  np.array(y).T
    z =  np.array(z).T
    c =  np.array(c).T

    return x, y, z, c

def get_M(dirname,smoothon=True):

    filename = dirname + '/dynamics/Mx.csv'
    Mx = read_csv(filename)

    filename = dirname + '/dynamics/My.csv'
    My = read_csv(filename)

    filename = dirname + '/dynamics/Mz.csv'
    Mz = read_csv(filename)

    Mx =  np.array(Mx).T
    My =  np.array(My).T
    Mz =  np.array(Mz).T

    if smoothon:
        for i in range(len(Mz)):
            Mx[i,:-1] = smooth(Mx[i,:],window_len=11)[1:-1]
            My[i,:-1] = smooth(My[i,:],window_len=11)[1:-1]
            Mz[i,:-1] = smooth(Mz[i,:],window_len=11)[1:-1]
    
    return Mx, My, Mz

def get_F(dirname,smoothon=True):
    filename = dirname + '/dynamics/Fx.csv'
    Fx = read_csv(filename)

    filename = dirname + '/dynamics/Fy.csv'
    Fy = read_csv(filename)

    filename = dirname + '/dynamics/Fz.csv'
    Fz = read_csv(filename)

    Fx =  np.array(Fx).T
    Fy =  np.array(Fy).T
    Fz =  np.array(Fz).T
        
    if smoothon:
        for i in range(len(Fz)):
            Fx[i,:-1] = smooth(Fx[i,:],window_len=11)[1:-1]
            Fy[i,:-1] = smooth(Fy[i,:],window_len=11)[1:-1]
            Fz[i,:-1] = smooth(Fz[i,:],window_len=11)[1:-1]

    return Fx, Fy, Fz

def read_whiskit_data(whiskers=['RA0'],pathin='',pathout='',parapath='',t0=0,tf=None):

    Mx,My,Mz = get_M(pathin)
    M = np.array([Mx,My,Mz])
    Fx,Fy,Fz = get_F(pathin)
    F = np.array([Fx,Fy,Fz])
    
    X = []
    Y = []
    Z = []
    C = []
    for w in whiskers:
        
        x,y,z,c = get_xyzc(pathin,w)
        X.append(x)
        Y.append(y)
        Z.append(z)          
        C.append(c)         

    C = np.array(C)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    
    if tf==None:
        tf = M.shape[2]

    M = M[:,:,t0:tf]
    F = F[:,:,t0:tf]
    X = X[:,:-1,t0:tf]
    Y = Y[:,:-1,t0:tf]
    Z = Z[:,:-1,t0:tf]
    C = C[:,:,t0:tf]

    if parapath != '':
        params = read_txt(parapath)
        print(params.shape)
    else:
        params=[]

    np.savez(pathout+'.npz',M=M,F=F,Col=C,X=X,Y=Y,Z=Z,simparam=params)
    sio.savemat(pathout+'.mat',dict(M=M,F=F,Col=C,X=X,Y=Y,Z=Z,simparam=params))
    print('Data loaded. Saved in: '+pathout)


    
