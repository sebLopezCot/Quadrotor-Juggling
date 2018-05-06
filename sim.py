
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle

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
        self.xmin = -5 #np.around(quad_states[:, 0].min() - self.cart_width / 2.0, 1)
        self.xmax = 5 #np.around(quad_states[:, 0].max() + self.cart_width / 2.0, 1)
        
        # create the axes
        self.ax = plt.axes(xlim=(self.xmin, self.xmax), ylim=(-5, 5), aspect='equal')
        
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
            #anim.save(filename, fps=30, codec='libx264')

        plt.show()

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


if __name__ == "__main__":

    quad = Quadrotor()
    ball = Ball()

    dt = 0.1
    steps = 1000

    time = dt * np.arange(steps)

    quad_state = np.zeros((steps, 6))
    quad_state[0,0] = 0.0 #np.random.choice(np.linspace(-5, 5, 20))
    quad_state[0,1] = 0.0 #np.random.choice(np.linspace(-12, 12, 20))
    quad_state[0,2] = 0.0 #np.random.choice(np.linspace(-np.pi/3.0, np.pi/3.0, 20))

    ball_state = np.zeros((steps, 4))
    ball_state[0, 0] = 0.0 #4.0
    ball_state[0, 1] = 10.0 #-1.0
    ball_state[0, 2] = 0.0 #-2.0
    ball_state[0, 3] = 0.0 #15.0

    lastErr_x = 0.0
    lastErr_y = 0.0
    lastErr_th = 0.0
    limit = 0.45
    desired_angle = 0.0

    for i in range(0,steps-1):
        contact = (np.linalg.norm(ball_state[i,0:2] - quad_state[i,0:2]) < 0.5)
        ball_state[i+1,:] = ball.step_dynamics(dt, ball_state[i,:], contact)
        error_x = quad_state[i,0] - 0.0
        dE_x = (error_x - lastErr_x) / dt
        limit = min(0.45 * abs(error_x)/5.0, 0.25)
        desired_angle = max(min(10000.0*error_x+ 10000.0*dE_x, limit), -limit) 
        error_y = 0.0 - quad_state[i,1]
        error_th = desired_angle - quad_state[i,2]
        dE_y = (error_y - lastErr_y) / dt
        dE_th = (error_th - lastErr_th) / dt
        force_command = 0.1*error_y + 0.3*dE_y + 0.981
        roll_command = 4.0*error_th + 1.5*dE_th
        input_u = np.array([force_command, roll_command])
        quad_state[i+1,:] = quad.step_dynamics(dt, quad_state[i,:], input_u)
        lastErr_y = error_y
        lastErr_th = error_th
        lastErr_x = error_x

    quad_state = np.reshape(quad_state[:,0:3], (steps, 3))

    a = Animator(time, quad_state, ball_state)





 
