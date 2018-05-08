
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy as sp

from pydrake.all import MathematicalProgram
from pydrake.symbolic import (sin, cos, tanh)

#def cos(x):
#    return 1.0 - (x**2.0)/2.0 + (x**4.0)/24.0 - (x**6.0)/720.0
#
#def sin(x):
#    return x - (x**3.0)/6.0 + (x**5.0)/120.0 - (x**7.0)/5040.0

def two_norm(x):
    slack = .001
    return (((x)**2).sum() + slack)**0.5

#def tanh(x):
#    return x - (x**3.0)/3.0 + 2.0*(x**5.0)/15.0 - 17.0*(x**7.0)/315.0

class BallQuadSystem(object):

    def __init__(self, quad_mass, g, restitution_coeff):
        self.quad_mass = quad_mass
        self.g = g
        self.g_vec = np.array([0.0, self.g])
        self.restitution_coeff = restitution_coeff

    def dynamics(self, quad_q, ball_q, quad_u, dt):
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

        # Ball
        ball_q_cp = np.copy(ball_q)
        
        # Collision
        epsilon = 1.0
        # Carefully, might hit singularity here of flipping indefinitely
        tan_comp = np.dot(ball_q_cp[2:4], tang) * tang
        norm_comp = np.dot(ball_q_cp[2:4], norm) * norm
        beta = self.restitution_coeff
        bounce_factor = tan_comp - beta * norm_comp # flipped the normal component
        dist_to_collision = two_norm(ball_q_cp[0:2] - quad_q[0:2])
        activation = tanh((1.0/epsilon)*dist_to_collision)
        ball_q_cp[2:4] = ball_q_cp[2:4] * activation + bounce_factor * (1.0 - activation)

        ball_next = np.zeros_like(ball_q)
        ball_next[0] = ball_q_cp[0] + ball_q_cp[2]*dt
        ball_next[1] = ball_q_cp[1] + ball_q_cp[3]*dt + 0.5*self.g*dt**2.0
        ball_next[2] = ball_q_cp[2] + 0.0
        ball_next[3] = ball_q_cp[3] + self.g*dt

        return quad_next, ball_next

    def solve(self, quad_start_q, quad_final_q, ball_start_q, ball_final_q, min_time, max_time):
        mp = MathematicalProgram()
        
        # We want to solve this for a certain number of knot points
        N = 100 # num knot points
        time_used = (max_time - min_time) / 2.0
        time_increment = time_used / (N+1)
        dt = time_increment
        time_array = np.arange(0.0, time_used, time_increment)

        quad_u = mp.NewContinuousVariables(2, "u_0")
        quad_q = mp.NewContinuousVariables(6, "quad_q_0")
        ball_q = mp.NewContinuousVariables(4, "ball_q_0")


        for i in range(1,N):
            u = mp.NewContinuousVariables(2, "u_%d" % i)        
            quad = mp.NewContinuousVariables(6, "quad_q_%d" % i)
            ball = mp.NewContinuousVariables(4, "ball_q_%d" % i)

            quad_u = np.vstack((quad_u, u))
            quad_q = np.vstack((quad_q, quad))
            ball_q = np.vstack((ball_q, ball))

        assert(quad_u.shape == (N, 2))
        assert(quad_q.shape == (N, 6))
        assert(ball_q.shape == (N, 4))

        for i in range(N):
            mp.AddLinearConstraint(quad_u[i][0] <= 100.0) # force
            mp.AddLinearConstraint(quad_u[i][0] >= 0.0) # force
            mp.AddLinearConstraint(quad_u[i][1] <= 100.0) # torque
            mp.AddLinearConstraint(quad_u[i][1] >= -100.0) # torque

            mp.AddLinearConstraint(quad_q[i][0] <= 1000.0) # pos x
            mp.AddLinearConstraint(quad_q[i][0] >= -1000.0)
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

            mp.AddLinearConstraint(ball_q[i][0] <= 1000.0) # pos x
            mp.AddLinearConstraint(ball_q[i][0] >= -1000.0)
            mp.AddLinearConstraint(ball_q[i][1] <= 1000.0) # pos y
            mp.AddLinearConstraint(ball_q[i][1] >= -1000.0)
            mp.AddLinearConstraint(ball_q[i][2] <= 100.0) # vel x
            mp.AddLinearConstraint(ball_q[i][2] >= -100.0)
            mp.AddLinearConstraint(ball_q[i][3] <= 100.0) # vel y
            mp.AddLinearConstraint(ball_q[i][3] >= -100.0)

        for i in range(1,N):
            quad_q_dyn_feasible, ball_q_dyn_feasible = self.dynamics(quad_q[i-1,:], ball_q[i-1,:], quad_u[i-1,:], dt)
            
            # Direct transcription constraints on states to dynamics
            for j in range(6):
                quad_state_err = (quad_q[i][j] - quad_q_dyn_feasible[j])
                eps = 0.01
                mp.AddConstraint(quad_state_err <= eps)
                mp.AddConstraint(quad_state_err >= -eps)

            for j in range(4):
                ball_state_err = (ball_q[i][j] - ball_q_dyn_feasible[j])
                eps = 0.01
                mp.AddConstraint(ball_state_err <= eps)
                mp.AddConstraint(ball_state_err >= -eps)


        # Initial and final quad and ball states
        for j in range(6):
            mp.AddLinearConstraint(quad_q[0][j] == quad_start_q[j])
            mp.AddLinearConstraint(quad_q[-1][j] == quad_final_q[j])
        
        for j in range(4):
            mp.AddLinearConstraint(ball_q[0][j] == ball_start_q[j])
            mp.AddLinearConstraint(ball_q[-1][j] == ball_final_q[j])

        # Quadratic cost on the control input
        R_force = 1.0
        R_torque = 100.0
        mp.AddQuadraticCost(R_force * quad_u[:,0].dot(quad_u[:,0]))
        mp.AddQuadraticCost(R_torque * quad_u[:,1].dot(quad_u[:,1]))

        # Solve the optimization
        print "Number of decision vars: ", mp.num_vars()

        print "Solve: ", mp.Solve()

        quad_traj = mp.GetSolution(quad_q)
        ball_traj = mp.GetSolution(ball_q)
        input_traj = mp.GetSolution(quad_u)

        return (quad_traj, ball_traj, input_traj, time_array)

class SystemVisualizer(object):

    def __init__(self):
        pass

    def visualize(self):
        pass

