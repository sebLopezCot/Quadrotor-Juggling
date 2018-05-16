
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy as sp

import matplotlib
from matplotlib.patches import Rectangle, Circle

from pydrake.all import MathematicalProgram, SolverType
from pydrake.symbolic import (sin, cos, tanh)

from IPython.display import HTML

def two_norm(x):
    slack = .001
    return (((x)**2).sum() + slack)**0.5

class BallQuadSystem(object):

    def __init__(self, quad_mass, ball_mass, g, restitution_coeff):
        self.quad_mass = quad_mass
        self.ball_mass = ball_mass
        self.g = g
        self.g_vec = np.array([0.0, self.g])
        self.restitution_coeff = restitution_coeff

    def dynamics(self, mp, contacts, last_dist, quad_q, ball_q, quad_u, dt):
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
        # activation = tanh((1.0/epsilon)*dist_to_collision)
        # ball_q_cp[2:4] = ball_q_cp[2:4] * activation + bounce_factor * (1.0 - activation)
        
        mp.AddConstraint(contacts[0] >= 0.0)
        mp.AddConstraint(dist_to_collision >= 0.0)
        mp.AddConstraint(dist_to_collision * contacts[0] == 0.0)

        J = (dist_to_collision - last_dist) / dt
        a_J = J / self.ball_mass
        a_J_comps = norm * a_J

        ball_next = np.zeros_like(ball_q)
        ball_next[0] = ball_q_cp[0] + ball_q_cp[2]*dt + 0.5*a_J_comps[0]*dt**2.0
        ball_next[1] = ball_q_cp[1] + ball_q_cp[3]*dt + 0.5*self.g*dt**2.0 + 0.5*a_J_comps[1]*dt**2.0
        ball_next[2] = ball_q_cp[2] + a_J_comps[0]*dt
        ball_next[3] = ball_q_cp[3] + self.g*dt + a_J_comps[1]*dt

        return quad_next, ball_next, dist_to_collision

    def solve(self, quad_start_q, quad_final_q, ball_start_q, ball_final_q, time_used):
        mp = MathematicalProgram()
        
        # We want to solve this for a certain number of knot points
        N = 100 # num knot points
        time_increment = time_used / (N+1)
        dt = time_increment
        time_array = np.arange(0.0, time_used, time_increment)

        quad_u = mp.NewContinuousVariables(2, "u_0")
        quad_q = mp.NewContinuousVariables(6, "quad_q_0")
        ball_q = mp.NewContinuousVariables(4, "ball_q_0")
        contacts = mp.NewContinuousVariables(1, "contacts_0")

        for i in range(1,N):
            u = mp.NewContinuousVariables(2, "u_%d" % i)        
            quad = mp.NewContinuousVariables(6, "quad_q_%d" % i)
            ball = mp.NewContinuousVariables(4, "ball_q_%d" % i)
            contact = mp.NewContinuousVariables(1, "contacts_%d" % i)

            quad_u = np.vstack((quad_u, u))
            quad_q = np.vstack((quad_q, quad))
            ball_q = np.vstack((ball_q, ball))
            contacts = np.vstack((contacts, contact))

        assert(quad_u.shape == (N, 2))
        assert(quad_q.shape == (N, 6))
        assert(ball_q.shape == (N, 4))
        assert(contacts.shape == (N, 1))

        for i in range(N):
            mp.AddLinearConstraint(quad_u[i][0] <= 100.0) # force
            mp.AddLinearConstraint(quad_u[i][0] >= 0.0) # force
            mp.AddLinearConstraint(quad_u[i][1] <= 100.0) # torque
            mp.AddLinearConstraint(quad_u[i][1] >= -100.0) # torque

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

            mp.AddLinearConstraint(ball_q[i][0] <= 1000.0) # pos x
            mp.AddLinearConstraint(ball_q[i][0] >= -1000.0)
            mp.AddLinearConstraint(ball_q[i][1] <= 1000.0) # pos y
            mp.AddLinearConstraint(ball_q[i][1] >= -1000.0)
            mp.AddLinearConstraint(ball_q[i][2] <= 100.0) # vel x
            mp.AddLinearConstraint(ball_q[i][2] >= -100.0)
            mp.AddLinearConstraint(ball_q[i][3] <= 100.0) # vel y
            mp.AddLinearConstraint(ball_q[i][3] >= -100.0)

        last_dist = 0.0

        for i in range(1,N):
            quad_q_dyn_feasible, ball_q_dyn_feasible, last_dist = self.dynamics(mp, contacts[i-1,:], last_dist, quad_q[i-1,:], ball_q[i-1,:], quad_u[i-1,:], dt)
            
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
            if j < 2:
                mp.AddLinearConstraint(ball_q[-1][j] == ball_final_q[j])
                

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

        mp.SetSolverOption(SolverType.kSnopt, "Major iterations limit", 100000)

        print "Solve: ", mp.Solve()

        quad_traj = mp.GetSolution(quad_q)
        ball_traj = mp.GetSolution(ball_q)
        input_traj = mp.GetSolution(quad_u)

        return (quad_traj, ball_traj, input_traj, time_array)

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
        self.restitution = 1.0

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


class Ball(object):

    def __init__(self):
        self.m = 0.001
        self.g = -9.81

    def step_dynamics(self, t, state, contact):
        if contact:
            state[3] = -state[3]*0.9
        
        state[0] += state[2]*t
        state[1] += state[3]*t + 0.5*self.g*t**2.0
        state[2] += 0.0
        state[3] += self.g*t

        return state

class Animator(object):

    def __init__(self, t, quad_states, ball_states):
        self.t = t
        self.quad_states = quad_states
        self.ball_states = ball_states
        
        # first set up the figure, the axis, and the plot elements we want to animate
        fig = plt.figure()
        
        # some dimesions
        self.cart_width = 0.4
        self.cart_height = 0.05
        
        # set the limits based on the motion
        self.xmin = -10 #np.around(quad_states[:, 0].min() - self.cart_width / 2.0, 1)
        self.xmax = 10 #np.around(quad_states[:, 0].max() + self.cart_width / 2.0, 1)
        
        # create the axes
        self.ax = plt.axes(xlim=(self.xmin, self.xmax), ylim=(-1, 10), aspect='equal')
        
        # display the current time
        self.time_text = self.ax.text(0.04, 0.9, '', transform=self.ax.transAxes)
        
        # create a rectangular cart
        self.ball = Rectangle([ball_states[0,0], ball_states[0,1]], 0.1, 0.1)
        self.rect = Rectangle([quad_states[0, 0] - self.cart_width / 2.0, quad_states[0,0]-self.cart_height / 2],
            self.cart_width, self.cart_height, fill=True, color='red', ec='black')
        self.rect_2 = Rectangle([quad_states[0,0] - self.cart_width / 2.0, quad_states[0,0]-self.cart_height / 2],
                0.05, 0.05, fill=True, color='blue', ec='black')
        self.rect_3 = Rectangle([quad_states[0,0] + self.cart_width / 2.0 - 0.05, quad_states[0,0]-self.cart_height / 2],
                0.05, 0.05, fill=True, color='blue', ec='black')
        self.ax.add_patch(self.rect)
        self.ax.add_patch(self.rect_2)
        self.ax.add_patch(self.rect_3)
        self.ax.add_patch(self.ball)

        # call the animator function
        self.anim = animation.FuncAnimation(fig, self.animate, frames=len(t), init_func=self.init_anim,
                interval=t[-1] / len(t) * 1000, blit=True, repeat=False)
        
        # save the animation if a filename is given
        #if filename is not None:
        self.anim.save('filename.mp4', fps=30, codec='libx264')

        #plt.show()

    def gen_html_animation(self):
        pass #HTML(self.anim.to_html5_video())

    def init_anim(self):
        self.time_text.set_text('')
        self.rect.set_xy((0.0, 0.0))
        self.rect_2.set_xy((0.0, 0.0))
        self.rect_3.set_xy((0.0, 0.0))
        self.ball.set_xy((0.0, 0.0))
        return self.time_text, self.rect, self.rect_2, self.rect_3, self.ball

    # animation function: update the objects
    def animate(self, i):
        self.time_text.set_text('time = {:2.2f}'.format(self.t[i]))
        
        self.ball.set_xy((self.ball_states[i,0], self.ball_states[i,1]))
        self.rect.set_xy((self.quad_states[i, 0] - self.cart_width / 2.0, self.quad_states[i,1]-self.cart_height / 2))
        self.rect_2.set_xy((self.quad_states[i,0] - self.cart_width / 2.0, self.quad_states[i,1]))
        self.rect_3.set_xy((self.quad_states[i,0] + self.cart_width / 2.0 - 0.05, self.quad_states[i,1]))

        t = matplotlib.transforms.Affine2D().rotate_around(self.quad_states[i, 0], self.quad_states[i, 1], self.quad_states[i, 2])

        self.rect.set_transform(t + plt.gca().transData)
        self.rect_2.set_transform(t + plt.gca().transData)
        self.rect_3.set_transform(t + plt.gca().transData)

        return self.time_text, self.rect, self.rect_2, self.rect_3, self.ball

