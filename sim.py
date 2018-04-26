
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

class Quadrotor(object):
    
    def __init__(self):
        self.m = 0.1
        self.l = 0.05
        self.g = -9.81
        self.max_f = 10.0
        self.min_f = 0.0
        self.max_roll = 1.0
        self.min_roll = -1.0
        self.restitution = 1.0

        self.state = np.vstack(np.zeros(6))

    def step(self, t, state):
        pass

    def draw(self, state):
        pass

#class Ball(object):
#
#    def __init__(self):
#        self.m = 0.001
#        self.

def animate_rocket(t, states, filename=None):

    # first set up the figure, the axis, and the plot elements we want to animate
    fig = plt.figure()
    
    # some dimesions
    cart_width = 0.4
    cart_height = 0.05
    
    # set the limits based on the motion
    xmin = np.around(states[:, 0].min() - cart_width / 2.0, 1)
    xmax = np.around(states[:, 0].max() + cart_width / 2.0, 1)
    
    # create the axes
    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 1.1), aspect='equal')
    
    # display the current time
    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)
    
    # create a rectangular cart
    rect = Rectangle([states[0, 0] - cart_width / 2.0, -cart_height / 2],
        cart_width, cart_height, fill=True, color='red', ec='black')
    rect_2 = Rectangle([states[0,0] - cart_width / 2.0, -cart_height / 2],
            0.05, 0.05, fill=True, color='blue', ec='black')
    rect_3 = Rectangle([states[0,0] + cart_width / 2.0 - 0.05, -cart_height / 2],
            0.05, 0.05, fill=True, color='blue', ec='black')
    ax.add_patch(rect)
    ax.add_patch(rect_2)
    ax.add_patch(rect_3)
    
    # initialization function: plot the background of each frame
    def init():
        time_text.set_text('')
        rect.set_xy((0.0, 0.0))
        rect_2.set_xy((0.0, 0.0))
        rect_3.set_xy((0.0, 0.0))
        return time_text, rect, rect_2, rect_3, 

    # animation function: update the objects
    def animate(i):
        time_text.set_text('time = {:2.2f}'.format(t[i]))
        rect.set_xy((states[i, 0] - cart_width / 2.0, -cart_height / 2))
        rect_2.set_xy((states[i,0] - cart_width / 2.0, 0))
        rect_3.set_xy((states[i,0] + cart_width / 2.0 - 0.05, 0))
        return time_text, rect, rect_2, rect_3, 

    # call the animator function
    anim = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init,
            interval=t[-1] / len(t) * 1000, blit=True, repeat=False)
    
    # save the animation if a filename is given
    #if filename is not None:
        #anim.save(filename, fps=30, codec='libx264')

    plt.show()


if __name__ == "__main__":

    time = 0.1 * np.arange(100)

    sin_t = np.sin((2*np.pi/3.0)*time)

    state = np.reshape(sin_t, (100, 1))

    animate_rocket(time, state, filename='test.mp4')







