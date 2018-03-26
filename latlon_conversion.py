# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:31:40 2018

@author: muhos601
"""
import math

def latlon_to_xy(lon,lat,R):
    """coverts latitude longitude to cartesian coordinate using area prserving Molleweide map projection"""
    if lon>180:
        lon=lon-360

    ilon = lon<0
    ilat = lat<0
    
    lat=abs(lat)/180*math.pi
    lon=abs(lon)/180*math.pi
    
    cm=0
    theta_p=lat
    delta_theta=1
    
    while(delta_theta>0.000005):
        delta_theta= -(theta_p+math.sin(theta_p)-math.pi*math.sin(lat))/(1+math.cos(theta_p))
        theta_p=theta_p+delta_theta
    
    theta=theta_p/2

    x=math.sqrt(8)/math.pi*R*(lon-cm)*math.cos(theta)
    y=math.sqrt(2)*R*math.sin(theta)
    
    if ilon==True:
        x = -1*x
    
    if ilat==True:
        y = -1*y
        
    return [x,y]
