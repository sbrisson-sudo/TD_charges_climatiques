# Importations

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np

import pyshtools   

import pickle

# Fonction de plot cartopy

def plot_cartopy_sh_grid(grid):
    
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12,6))
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')

    # c = ax.contourf(grid3.lons(), grid3.lats(), grid3.data, transform=ccrs.PlateCarree(), cmap='viridis')

    lon,lat = grid.lons(), grid.lats()
    c = ax.imshow(grid.data, extent=[lon.min(),lon.max(),lat.min(),lat.max()], transform=ccrs.PlateCarree(), cmap='bwr')

    cbar = plt.colorbar(c, ax=ax, orientation='vertical', label='', shrink=0.5)
    
    return fig,ax,cbar

def plot_cartopy_Clm(Clm):
    
    Clm_sh = pyshtools.SHCoeffs.from_array(Clm)
    grid = Clm_sh.expand()
    
    return plot_cartopy_sh_grid(grid)

def plot_cartopy(lat,lon,data,cbar_label=""):
    
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12,6))
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')

    # c = ax.contourf(grid3.lons(), grid3.lats(), grid3.data, transform=ccrs.PlateCarree(), cmap='viridis')

    lon,lat = lon,lat
    c = ax.imshow(data, extent=[lon.min(),lon.max(),lat.min(),lat.max()], transform=ccrs.PlateCarree(), cmap='bwr')

    cbar = plt.colorbar(c, ax=ax, orientation='vertical', label=cbar_label, shrink=0.5)
    
    return fig,ax,cbar


# Lecture données

def read_grace_file(file):
    
    (l, m, C, S) = np.loadtxt(file, skiprows=125, usecols=range(1, 5), dtype='i,i,d,d', unpack=True)

    lmax = np.max(l)

    Clm = np.zeros((2, lmax+1, lmax+1))
    for i in range(len(l)):
        Clm[0, l[i], m[i]] = C[i]
        Clm[1, l[i], m[i]] = S[i]
                
    return Clm

def read_grace_file_anomaly(file, Clm0):
    
    (l, m, C, S) = np.loadtxt(file, skiprows=125, usecols=range(1, 5), dtype='i,i,d,d', unpack=True)

    lmax = np.max(l)

    dClm = np.zeros((2, lmax+1, lmax+1))
    for i in range(len(l)):
        if (l[i] <= lmax):
            dClm[0, l[i], m[i]] = Clm0[0, l[i], m[i]] - C[i]
            dClm[1, l[i], m[i]] = Clm0[1, l[i], m[i]] - S[i]
                    
    return dClm

W_DDK5 = pickle.load(open('DDK5.pickle', 'rb'))

def filter_DDK5(Clm):
    """Applique filtre DDK5"""
    
    lmax = Clm.shape[1]-1

    Clm[0,2:lmax+1,0] = np.dot(W_DDK5[0,0].T[:lmax-1,:lmax-1], Clm[0,2:lmax+1,0])
    Clm[0,2:lmax+1,1] = np.dot(W_DDK5[0,1].T[:lmax-1,:lmax-1], Clm[0,2:lmax+1,1])
    Clm[1,2:lmax+1,1] = np.dot(W_DDK5[1,1].T[:lmax-1,:lmax-1], Clm[1,2:lmax+1,1])
    for m in range(2, lmax+1):
        Clm[0,m:lmax+1,m] = np.dot(W_DDK5[0,m].T[:lmax+1-m,:lmax+1-m], Clm[0,m:lmax+1,m])
        Clm[1,m:lmax+1,m] = np.dot(W_DDK5[1,m].T[:lmax+1-m,:lmax+1-m], Clm[1,m:lmax+1,m])
        
    return Clm

# Passage en hauteur d'eau équivalente

ae = 6371000. # [m]
rhoe = 5514.  # [kg/m^3]
rhow = 1000.  # [kg/m^3]
(hlove, llove, klove) = np.loadtxt('Load_Love2_CF.dat', usecols=range(1, 4), unpack=True)

def dClm_to_Hlm(dClm):
    
    Hlm = np.zeros(dClm.shape)
    
    for l in range(1, dClm.shape[1]):
        Hlm[:,l,:] = rhoe*ae*(2*l+1) / (3*rhow*(1+klove[l])) * dClm[:,l,:]

    return pyshtools.SHCoeffs.from_array(Hlm)

# Passage à la déformation au sol

def dClm_to_UN1N2Elm(dClm):
    
    Ulm,N1lm,N2lm,Elm = np.zeros(dClm.shape),np.zeros(dClm.shape),np.zeros(dClm.shape),np.zeros(dClm.shape)
    
    for l in range(1, dClm.shape[1]):
        
        Ulm[:,l,:] = ae*hlove[l]/(1+klove[l]) * dClm[:,l,:]
        
        for m in range(l+1):
            
            N1lm[:,l,m] = -ae*llove[l]/(1+klove[l])*l * dClm[:,l,m]
            N2lm[:,l-1,m] = ae*llove[l]/(1+klove[l])*np.sqrt(float(2*l+1)/(2*l-1)*(l+m)*(l-m)) * dClm[:,l,m]
            
            Elm[0,l,m] =  ae*llove[l]/(1+klove[l])*m * dClm[1,l,m]
            Elm[1,l,m] = -ae*llove[l]/(1+klove[l])*m * dClm[0,l,m]
            
    return map(pyshtools.SHCoeffs.from_array,[Ulm,N1lm,N2lm,Elm])

# Extrapolation en champ de deformation

