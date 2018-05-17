
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
import scipy.interpolate

from quad_direct_transcription import QuadDirectTranscription

class TimeInterpolator(object):

    def __init__(self, time_arr, traj, interp_steps, dt, start_time=0):
        print "Constructing time interpolator..."

        self.interps = [scipy.interpolate.interp1d(time_arr[:-1], traj[:,i]) for i in range(traj.shape[1])]
        self.cutoff_index = np.inf

        q = len(self.interps)
        self.interp_traj = np.zeros((interp_steps, q))

        loophalted = False
        for i in range(int(start_time/dt), interp_steps):
            if i*dt < start_time:
                continue
            for j in range(q):
                try:
                    self.interp_traj[i,j] = self.interps[j](i*dt - start_time)
                except:
                    self.cutoff_index = i
                    loophalted = True
                    break

            if loophalted:
                break

        print "Done constructing time interpolator!"

    def __getitem__(self, i):
        return self.interp_traj[i,:]

    def get_cutoff_index(self):
        return self.cutoff_index

class ProjectileMath(object):

    @staticmethod
    def calc_arch_info(start_of_rainbow):
        theta = np.sign(start_of_rainbow[2]) * np.arctan(abs(start_of_rainbow[3]) / abs(start_of_rainbow[2])) if abs(start_of_rainbow[2]) > 1e-6 else 0.0
        norm = np.array([np.cos(theta), np.sin(theta)])
        vlaunch = norm * start_of_rainbow[3] / np.sin(theta) if abs(np.sin(theta)) > 1e-6 else np.zeros(2)
        vland = vlaunch
        vland[1] *= -1.0
        rainbow_width = 2.0*np.linalg.norm(vlaunch)**2.0 * np.sin(abs(theta)) * np.cos(theta) / 9.81
        rainbow_height = start_of_rainbow[3]**2.0 / 2.0 / 9.81
        travel_time = rainbow_width / (abs(start_of_rainbow[2])) if abs(start_of_rainbow[2]) > 1e-6 else 0.0

        return (rainbow_width, rainbow_height, travel_time, vland, theta)

    @staticmethod
    def get_drop_time(state):
        return np.sqrt(2.0 * state[1] / 9.81)

class ContactingSwitch(object):

    def __init__(self):
        self.STATE_NO_CONTACT = 0
        self.STATE_FIRST_CONTACT = 1
        self.STATE_CONTACTING = 2

        self.state = self.STATE_NO_CONTACT

    def leads_to_contact(self, contact):
        if contact and self.state == self.STATE_NO_CONTACT:
            self.state = self.STATE_FIRST_CONTACT
        elif contact and self.state == self.STATE_CONTACTING:
            pass
        elif contact and self.state == self.STATE_FIRST_CONTACT:
            self.state = self.STATE_CONTACTING
        elif not contact and self.state == self.STATE_NO_CONTACT:
            pass
        elif not contact and self.state == self.STATE_CONTACTING:
            self.state = self.STATE_NO_CONTACT
        elif not contact and self.state == self.STATE_FIRST_CONTACT:
            self.state = self.STATE_NO_CONTACT

        return self.state == self.STATE_FIRST_CONTACT

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


class Ball(object):

    def __init__(self):
        self.m = 0.001
        self.g = -9.81
        self.restitution = 0.6

    def step_dynamics(self, t, state, quad_state, contact, contact_dir):
        if contact:
            print state[2:4]
            tang = np.array([[0.0, 1.0], [-1.0, 0.0]]).dot(contact_dir)
            tang_comp = tang * state[2:4].dot(tang) if abs(np.linalg.norm(state[2:4])) > 1e-6 else np.zeros(2)
            print "tang_comp", tang_comp
            norm_comp = contact_dir * state[2:4].dot(contact_dir) if abs(np.linalg.norm(state[2:4])) > 1e-6 else np.zeros(2)
            print "norm_comp", norm_comp
            state[2:4] = tang_comp - norm_comp*self.restitution + quad_state[3:5]
            print state[2:4]
        
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
        self.xmin = -20 #np.around(quad_states[:, 0].min() - self.cart_width / 2.0, 1)
        self.xmax = 10 #np.around(quad_states[:, 0].max() + self.cart_width / 2.0, 1)
        
        # create the axes
        self.ax = plt.axes(xlim=(self.xmin, self.xmax), ylim=(-5, 10), aspect='equal')
        
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
        anim = animation.FuncAnimation(fig, self.animate, frames=len(t), init_func=self.init_anim,
                interval=t[-1] / len(t) * 1000, blit=True, repeat=False)
        
        # save the animation if a filename is given
        #if filename is not None:
        anim.save('filename.mp4', fps=30, codec='libx264')

        # plt.show()

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

