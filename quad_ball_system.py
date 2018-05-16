
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy as sp

from pydrake.all import MathematicalProgram, SolverType
from pydrake.symbolic import (sin, cos, tanh)

from IPython.display import HTML

class QuadDirectTranscription(object):

    def __init__(self, quad_mass, g):
        self.quad_mass = quad_mass
        self.g = g
        self.g_vec = np.array([0.0, self.g])

    def dynamics(self, quad_q, quad_u, dt):
        # Quadrotor
        a_f = quad_u[0] * 1.0 / self.quad_mass
        r_ddot = quad_u[1]
        tang = np.array([cos(quad_q[2]), sin(quad_q[2])])
        norm = np.array([-sin(quad_q[2]), cos(quad_q[2])])
        pos_ddot = norm * a_f + self.g_vec

        quad_next = np.zeros_like(quad_q)
        quad_next[0] = quad_q[0] + quad_q[3]*dt + 0.5*pos_ddot[0]*dt**2.0
        quad_next[1] = quad_q[1] + quad_q[4]*dt + 0.5*pos_ddot[1]*dt**2.0
        quad_next[2] = quad_q[2] + quad_q[5]*dt + 0.5*r_ddot*dt**2.0
        quad_next[3] = quad_q[3] + pos_ddot[0]*dt
        quad_next[4] = quad_q[4] + pos_ddot[1]*dt
        quad_next[5] = quad_q[5] + r_ddot*dt

        return quad_next

    def solve(self, quad_start_q, quad_final_q, time_used):
        mp = MathematicalProgram()
        
        # We want to solve this for a certain number of knot points
        N = 20 # num knot points
        time_increment = time_used / (N+1)
        dt = time_increment
        time_array = np.arange(0.0, time_used, time_increment)

        quad_u = mp.NewContinuousVariables(2, "u_0")
        quad_q = mp.NewContinuousVariables(6, "quad_q_0")

        for i in range(1,N):
            u = mp.NewContinuousVariables(2, "u_%d" % i)        
            quad = mp.NewContinuousVariables(6, "quad_q_%d" % i)

            quad_u = np.vstack((quad_u, u))
            quad_q = np.vstack((quad_q, quad))

        for i in range(N):
            mp.AddLinearConstraint(quad_u[i][0] <= 3.0) # force
            mp.AddLinearConstraint(quad_u[i][0] >= 0.0) # force
            mp.AddLinearConstraint(quad_u[i][1] <= 10.0) # torque
            mp.AddLinearConstraint(quad_u[i][1] >= -10.0) # torque

            mp.AddLinearConstraint(quad_q[i][0] <= 100.0) # pos x
            mp.AddLinearConstraint(quad_q[i][0] >= -100.0)
            mp.AddLinearConstraint(quad_q[i][1] <= 1000.0) # pos y
            mp.AddLinearConstraint(quad_q[i][1] >= -1000.0)
            mp.AddLinearConstraint(quad_q[i][2] <= 60.0 * np.pi / 180.0) # pos theta
            mp.AddLinearConstraint(quad_q[i][2] >= -60.0 * np.pi / 180.0)
            mp.AddLinearConstraint(quad_q[i][3] <= 100.0) # vel x
            mp.AddLinearConstraint(quad_q[i][3] >= -100.0)
            mp.AddLinearConstraint(quad_q[i][4] <= 100.0) # vel y
            mp.AddLinearConstraint(quad_q[i][4] >= -100.0)
            mp.AddLinearConstraint(quad_q[i][5] <= 10.0) # vel theta
            mp.AddLinearConstraint(quad_q[i][5] >= -10.0)

        for i in range(1,N):
            quad_q_dyn_feasible = self.dynamics(quad_q[i-1,:], quad_u[i-1,:], dt)
            
            # Direct transcription constraints on states to dynamics
            for j in range(6):
                quad_state_err = (quad_q[i][j] - quad_q_dyn_feasible[j])
                eps = 0.01
                mp.AddConstraint(quad_state_err <= eps)
                mp.AddConstraint(quad_state_err >= -eps)

        # Initial and final quad and ball states
        for j in range(6):
            mp.AddLinearConstraint(quad_q[0][j] == quad_start_q[j])
            mp.AddLinearConstraint(quad_q[-1][j] == quad_final_q[j])
        
        # Quadratic cost on the control input
        R_force = 1.0
        R_torque = 100.0
        Q_quad_x = 100.0
        Q_quad_y = 100.0
        mp.AddQuadraticCost(R_force * quad_u[:,0].dot(quad_u[:,0]))
        mp.AddQuadraticCost(R_torque * quad_u[:,1].dot(quad_u[:,1]))
        mp.AddQuadraticCost(Q_quad_x * quad_q[:,0].dot(quad_q[:,1]))
        mp.AddQuadraticCost(Q_quad_y * quad_q[:,1].dot(quad_q[:,1]))

        # Solve the optimization
        print "Number of decision vars: ", mp.num_vars()

        # mp.SetSolverOption(SolverType.kSnopt, "Major iterations limit", 100000)

        print "Solve: ", mp.Solve()

        quad_traj = mp.GetSolution(quad_q)
        input_traj = mp.GetSolution(quad_u)

        return (quad_traj, input_traj, time_array)

class Quadrotor(object):
    
    def __init__(self):
        self.m = 0.1
        self.l = 0.05
        self.g = -9.81
        self.g_vec = np.array([0.0, self.g])
        self.max_f = 10.0
        self.min_f = 0.0
        self.max_roll = 1.0
        self.min_roll = -1.0

    def step_dynamics(self, t, state, u):
        a_f = u[0] * 1.0 / self.m
        r_ddot = u[1]
        norm = np.array([-np.sin(state[2]), np.cos(state[2])])
        pos_ddot = norm * a_f + self.g_vec 
        
        state[0] += state[3]*t + 0.5*pos_ddot[0]*t**2.0
        state[1] += state[4]*t + 0.5*pos_ddot[1]*t**2.0
        state[2] += state[5]*t + 0.5*r_ddot*t**2.0
        state[3] += pos_ddot[0]*t
        state[4] += pos_ddot[1]*t
        state[5] += r_ddot*t

        return state
