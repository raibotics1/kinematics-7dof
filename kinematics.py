#Fk an IK Kinematics for 7DoF Panda Emika



import numpy as np
from scipy.optimize import fmin
from math import *

joints= np.array([-4.91426098,  7.30237574,  0.06428148,  3.33337267,  0.71807756, -3.62860267,-5.87764174])
target_position=np.array([0,0,4])

# Function for Forward Kinematics


def forward_kinematics(joints):
    q = joints
    DH = np.array([
        [0,  0.333,  0,      0],
        [0,  0,     -pi/2,   0],
        [0,  0.316,  pi/2,   0],
        [0,  0,      pi/2,   0.0825],
        [0,  0.384,  -pi/2, -0.0825],
        [0,  0,      pi/2,   0], 
        [0,  0.107,  pi/2,   0.088]
    ])

    T= np.eye(4)

    for i in range(len(DH)):
       T = np.dot(T, np.array([
            [np.cos(q[i]), -np.sin(q[i]), 0, DH[i,3]],
            [np.sin(q[i])   *   np.cos(DH[i,2]), np.cos(q[i])   *   np.cos(DH[i,2]), -np.sin(DH[i,2]), -np.sin(DH[i,2])   *   DH[i,1]],
            [np.sin(q[i])   *   np.sin(DH[i,2]), np.cos(q[i])   *   np.sin(DH[i,2]), np.cos(DH[i,2]), np.cos(DH[i,2])   *   DH[i,1]],
            [0, 0, 0, 1],
        ]))
    position=[T[0,3],T[1,3],T[2,3]]
    # Return Manipulator position
    return T,position

def jacobian(joints):
    # Function for Jacobian's Matrix
    DH = np.array([
        [0,  0.333,  0,      0],
        [0,  0,     -pi/2,   0],
        [0,  0.316,  pi/2,   0],
        [0,  0,      pi/2,   0.0825],
        [0,  0.384,  -pi/2, -0.0825],
        [0,  0,      pi/2,   0], 
        [0,  0.107,  pi/2,   0.088]
    ])

    T, _ = forward_kinematics(joints)
    rotMat = T[:3, :3]
    linearJointVelocities = np.dot(rotMat, np.transpose(DH[:, 3]))
    jacobianMatrix = np.concatenate((linearJointVelocities, rotMat), axis=1)
    return jacobianMatrix

def residual(joints):
    T, position = forward_kinematics(joints)
    res=np.linalg.norm(target_position-position)
    return res



def inverse_kinematics(target_position):
    # Position 0
    initialGuess = np.zeros(7)
    result = fmin(residual, initialGuess, disp=False)
    return result




fK=forward_kinematics(joints)
iK = inverse_kinematics(target_position)


print("Forward Kinematics", fK)
print("Inverse kinematics:", iK)
