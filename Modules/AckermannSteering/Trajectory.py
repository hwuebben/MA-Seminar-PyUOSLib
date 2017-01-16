# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:57:40 2015

@author: JSCHOENK
"""
from __future__ import division

import numpy as np
import pylab as plt
import scipy.optimize as optimize

try:
    from MPyUOSLib import BasicProcessingModule
except ImportError:
    class BasicProcessingModule:
        def __init__(self, foot):
            pass


def getCoefDiff0(t):
    return np.array([t**3,t**2,t,1])

def getCoefDiff1(t):
    return np.array([3*t**2,2*t,1,0])

def getCoefDiff2(t):
    return np.array([6*t,2,0,0])

def calcCoef(values):
    matDim = 4 * len(values)
    coef = np.zeros([matDim,matDim])
    b = np.zeros(matDim)
    coef[0,0:4] = getCoefDiff0(0)
    coef[1,-4:] = getCoefDiff0(1)
    coef[2,0:4] = getCoefDiff1(0)
    coef[2,-4:] = -getCoefDiff1(1)
    coef[3,0:4] = getCoefDiff2(0)
    coef[3,-4:] = -getCoefDiff2(1)
    b[0] = values[0]
    b[1] = values[0]
    for i in xrange(1, len(values)):
        coef[4*i  ,4*i:4*(i+1)] = getCoefDiff0(0)        
        coef[4*i+1,4*(i-1):4*i] = getCoefDiff0(1)
        coef[4*i+2,4*i:4*(i+1)] = getCoefDiff1(0)
        coef[4*i+2,4*(i-1):4*i] = -getCoefDiff1(1)
        coef[4*i+3,4*i:4*(i+1)] = getCoefDiff2(0)
        coef[4*i+3,4*(i-1):4*i] = -getCoefDiff2(1)
        b[4*i+0] = values[i]
        b[4*i+1] = values[i]
    inv = np.linalg.inv(coef)
    return np.dot(inv, b).reshape(len(values), 4)


class Localization(BasicProcessingModule):
    """Localization of car on a trajectory based on pose (x,y,phi)
    
    x - Position on x-axis
    y - Position on y-axis
    phi - Orientation
    
    This module localizes a car on a trajectory given by its pose (x,y,phi) and
    calculate the distance and misalignment to the trajectory at the look-ahead 
    distance.
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        
        self.timeEstimation = 1.0
        self.points = np.array(foot["points"])
        if len(self.points) == 2:
            self.radius = np.linalg.norm(self.points[0] - self.points[1]) / 2
            self.center = np.mean(self.points, axis=0)
        else:
            self.coefX = calcCoef(self.points.T[0,:])
            self.coefY = calcCoef(self.points.T[1,:])
        try:
            self.look_ahead = np.array(foot["lookAhead"])
        except KeyError:
            self.look_ahead = 1.0
            
        self.output = np.array([0,0,0])

    def prepare(self, antecessor):
        startPose = antecessor["pose"].output
        if np.size(startPose) != 3:
            startPose = np.array([0,0,0])
        
        start_x = startPose[0]
        start_y = startPose[1]
        start_phi = startPose[2]
        self.timeEstimation = self.initialLocalization(start_x, start_y)
        self.timeEstimation = self.relocalize(start_x, start_y, self.timeEstimation)
        
        pos_est = self.evalTrajectory(self.timeEstimation)
        delta_x = pos_est[0] - start_x
        delta_y = pos_est[1] - start_y
        error_phi = self.getMisalignment(self.timeEstimation, start_phi)
        self.output = np.array([delta_x, delta_y, error_phi])

    def __call__(self, pose, lookAheadIn=None, index=0):
        if lookAheadIn:
            self.look_ahead = lookAheadIn

        current_x = pose[0]
        current_y = pose[1]
        current_phi = pose[2]
        self.timeEstimation = self.relocalize(current_x, current_y, self.timeEstimation)
        self.timeEstimation_la = self.getLookAheadTime(self.timeEstimation, self.look_ahead)
        
 #       pos_est = self.evalTrajectory(self.timeEstimation)
 #       delta_vec = np.array([pos_est[0] - current_x, pos_est[1] - current_y])
 #       error_phi = self.getMisalignment(self.timeEstimation, current_phi)     
        
        pos_est = self.evalTrajectory(self.timeEstimation_la)     
        delta_vec = np.array([pos_est[0] - current_x, pos_est[1] - current_y])
        delta_vec_rot = np.dot(self.getRotMat(np.radians(90) + current_phi), delta_vec)
        error_phi = self.getMisalignment(self.timeEstimation_la, current_phi)

        return np.array([delta_vec_rot[0], delta_vec_rot[1], error_phi])
                         
    def getRotMat(self, phi):
        return np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
        
    def evalTrajectory(self, time):
        localTime, index = self.getSegmentTimeAndIndex(time)
        if len(self.points) == 2:
            if np.mod(time, 1) < 0.5:
                x_t = self.radius * -np.cos(np.pi * localTime) + self.center[0]
                y_t = self.radius * np.sin(np.pi * localTime) + self.center[1]
            else:
                x_t = self.radius * np.cos(np.pi * localTime) + self.center[0]
                y_t = self.radius * -np.sin(np.pi * localTime) + self.center[1]
        else:
            x_t = np.polyval(self.coefX[index], localTime)
            y_t = np.polyval(self.coefY[index], localTime)
        return np.array([x_t, y_t])
        
    def evalTangentVector(self, time):
        localTime, index = self.getSegmentTimeAndIndex(time)
        if len(self.points) == 2:
            if np.mod(time, 1) < 0.5:
                x_t = self.radius * -np.cos(np.pi * localTime) + self.center[0]
                y_t = self.radius * np.sin(np.pi * localTime) + self.center[1]
            else:
                x_t = self.radius * np.cos(np.pi * localTime) + self.center[0]
                y_t = self.radius * -np.sin(np.pi * localTime) + self.center[1]
        else:
            x_t = np.inner(self.coefX[index], getCoefDiff1(localTime))
            y_t = np.inner(self.coefY[index], getCoefDiff1(localTime))
        return np.array([x_t, y_t])
        
    def evalTrajectoryPhi(self, time):
        tangent = self.evalTangentVector(time)
        return np.arctan2(tangent[1], tangent[0])
    
    def getSegmentTimeAndIndex(self, time):
        time = np.mod(time, 1)
        totalTime = time * len(self.points)
        index = int(np.floor(totalTime))
        localTime = totalTime - index
        return localTime, index
    
    def getLookAheadTime(self, time, distance):       
        point_a = self.evalTrajectory(time)
        delta_t = 1.0 / len(self.points)        
        time_b = time + delta_t
        point_b = self.evalTrajectory(time_b)
        d = np.linalg.norm(point_a - point_b)
        
        if d < distance:
            while d < distance:
                time_b += delta_t
                delta_t *= 2
                point_b = self.evalTrajectory(time_b)
                d = np.linalg.norm(point_a - point_b)
                time_a = time_b - delta_t / 2
        else:
            time_a = time
        
        for i in range(10):
            pivot = (time_a + time_b) / 2
            point_b = self.evalTrajectory(pivot)
            d = np.linalg.norm(point_a - point_b)
            if d < distance:
                time_a = pivot
            else:
                time_b = pivot
        
        return (time_a + time_b) / 2
    
    def evalTrajectorySegment(self, time, index):
        if len(self.points) == 2:
            if index == 0:
                x_t = self.radius * -np.cos(np.pi * time) + self.center[0]
                y_t = self.radius * np.sin(np.pi * time) + self.center[1]
            else:
                x_t = self.radius * np.cos(np.pi * time) + self.center[0]
                y_t = self.radius * -np.sin(np.pi * time) + self.center[1]
        else:
            x_t = np.polyval(self.coefX[index], time)
            y_t = np.polyval(self.coefY[index], time)
        return np.array([x_t,y_t])
        
    def getLocalLinearTime(self, x, y, index):
        if index == len(self.points) - 1:
            x0 = self.points[-1][0]
            x1 = self.points[ 0][0]
            y0 = self.points[-1][1]
            y1 = self.points[ 0][1]
        else:
            x0 = self.points[index    ][0]
            x1 = self.points[index + 1][0]
            y0 = self.points[index    ][1]
            y1 = self.points[index + 1][1]
        t = ((x-x0) * (x1-x0) + (y-y0) * (y1-y0)) / ((x0-x1)**2 + (y0-y1)**2)
        return t
        
    def initialLocalization(self, x, y):
        minDist = np.float('inf')
        best_index = 0
        best_t = 0
        for i in range(len(self.points)):
            t = self.getLocalLinearTime(x, y, i)
            if (0 <= t) and (t <= 1):
                dist = self.getDistanceSegment(t, x, y, i)
                if dist<minDist:
                    minDist = dist
                    best_index = i
                    best_t = t        
        return (best_index + best_t) / len(self.points)
    
    def getDistance(self, t, x, y):
        trajectory_position = self.evalTrajectory(t)
        distX = (x - trajectory_position[0])**2
        distY = (y - trajectory_position[1])**2
        return np.sqrt(distX + distY)
        
    def getDistanceSegment(self, t, x, y, index):
        trajectory_position = self.evalTrajectorySegment(t, index)
        distX = (x - trajectory_position[0])**2
        distY = (y - trajectory_position[1])**2
        return np.sqrt(distX + distY)
        
    def getMisalignment(self, t, phi):
        trajectory_phi = self.evalTrajectoryPhi(t)
        delta_phi = trajectory_phi - phi
        delta_phi = np.mod(delta_phi + np.pi, 2 * np.pi) - np.pi
        return delta_phi
        
    def localize(self, x, y, t0):
        delta = 1 / (4 * len(self.points))
        t_a = t0
        t_b = t0 + delta
        t_opt = optimize.brent(self.getDistance, args=(x,y), brack=(t_a,t_b))
        return t_opt
    
    def localizeInSegment(self, x, y, t0, index):
        t_a = 0.0
        t_b = 1.0
        t_opt = optimize.brent(self.getDistanceSegment, args=(x,y,index), brack=(t_a,t_b))
        
        return t_opt
        
    def relocalize(self, x, y, t0):
        t0 = self.timeEstimation
        localTime, index = self.getSegmentTimeAndIndex(t0)
        local_est_time = self.getLocalLinearTime(x, y, index)
        if local_est_time > localTime:
            localTime = local_est_time
        globalTime = (index + localTime) / len(self.points)
        localTime, index = self.getSegmentTimeAndIndex(globalTime)
        local_est_time = self.localizeInSegment(x, y, localTime, index)
        brent_time = self.localize(x, y, t0)
        brentLocalTime, brentIndex = self.getSegmentTimeAndIndex(brent_time)
        if index == brentIndex:
            local_est_time = brentLocalTime
        new_est_time = (index + local_est_time) / len(self.points)
        if new_est_time > t0:
            return new_est_time
        elif new_est_time < 1 / len(self.points):
            return new_est_time
        else:
            return t0
    

class TrackingError(BasicProcessingModule, Localization):
    """Calculation of the tracking error
    
    This module localizes a car on a trajectory given by its pose (x,y,phi) and
    calculate the distance and misalignment to the trajectory at the current 
    position.
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        Localization.__init__(self, foot)
        self.carTime = 1.0
        self.estTime = 1.0
        self.output = np.array([0,0])
    
    def prepare(self, antecessor):
        carPose = antecessor["car"].output
        self.carTime = self.initialLocalization(carPose[0], carPose[1])
        carPose_traj = self.evalTrajectory(self.carTime)    
        car_dist = np.linalg.norm(carPose_traj - carPose[:2])
        
        try:
            estPose = antecessor["estimate"].output
            self.estTime = self.initialLocalization(estPose[0], carPose[1])
            estPose_traj = self.evalTrajectory(self.estTime)
            est_dist = np.linalg.norm(estPose_traj - estPose[:2])
            self.output = np.array([car_dist, est_dist])
        except KeyError:
            self.output = np.array([car_dist])
    
    def __call__(self, index=0, **argIn):
        carPose = argIn["car"]
        self.carTime = self.relocalize(carPose[0], carPose[1], self.carTime)    
        carPose_traj = self.evalTrajectory(self.carTime)  
        car_dist = np.linalg.norm(carPose_traj - carPose[:2])
        
        try:
            estPose = argIn["estimate"]
            self.estTime = self.relocalize(estPose[0], estPose[1], self.estTime) 
            estPose_traj = self.evalTrajectory(self.estTime)   
            est_dist = np.linalg.norm(estPose_traj - estPose[:2])
            return np.array([car_dist, est_dist])
        except KeyError:
            return np.array([car_dist])


class AverageTrackingError(BasicProcessingModule):
    """Calculation of the average tracking error
    
    This module calculates the Euclidean distance between a given and the actual 
    driven trajectory at each measurement and return the average error over the 
    entire process.      
    """
    def __init__(self, foot):
        BasicProcessingModule.__init__(self, foot)
        self.dist = 0
        self.avg = 0
        
    def __call__(self, error, index=0):
        self.dist += error
        self.avg = self.dist / (index + 1)
        return self.avg

        
if __name__=="__main__":
    points = np.array([[4,-6],[0,-2],[0,12],[4,16],[12,14],[8,12],[4,10],[8,8],[12,6],[8,4],[4,0],[8,-4],[10,-2],[9,-2],[7,0],[9,1],[12,-4]])
    a = Localization({"points":points})
    coefX = calcCoef(points.T[0,:])
    coefY = calcCoef(points.T[1,:])
    t = np.linspace(0, 1, 31)
    plt.plot(points.T[0,:], points.T[1,:], 'ok')
    trajP = []
    trajT = []
    for time in t:
        point = a.evalTrajectory(time)
        tangent = a.evalTangentVector(time)
        trajP.append(point)
        trajT.append(tangent)
        plt.plot([point[0],point[0]+tangent[0]], [point[1],point[1]+tangent[1]])
    trajP = np.array(trajP)
    plt.plot(trajP.T[0,:], trajP.T[1,:], 'rx')
    for i in range(len(points.T[0,:])):
        x1 = np.polyval(coefX[i], t)
        y1 = np.polyval(coefY[i], t)
        plt.plot(x1, y1)
    
    x_init = 10
    y_init = -4
    t_init = a.initialLocalization(x_init, y_init)
    est_init = a.evalTrajectory(t_init)
    est_init_opt = a.evalTrajectory(a.localize(x_init, y_init, t_init))
    plt.plot([x_init,est_init[0],est_init_opt[0]], [y_init,est_init[1],est_init_opt[1]], 'or-')
    plt.axis('equal')
#    plt.xlim([-1,3])
#    plt.ylim([-1,3])
    plt.show()