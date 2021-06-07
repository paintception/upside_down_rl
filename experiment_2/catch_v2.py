import random
import numpy as np
from scipy.misc import imresize

import random
import numpy as np
from scipy.ndimage import rotate
from scipy.misc import imresize
from matplotlib import pyplot as plt 

class CatchEnv2:
    def __init__(self):
        self.size = 21
        self.image = np.zeros((self.size, self.size))
        self.state = []
        self.fps = 4
        self.output_shape = (84, 84)

    def reset_random(self):
        self.image.fill(0)
        self.pos = np.random.randint(2, self.size-2)
        self.vx = np.random.randint(5) - 2
        self.vy = 1
        self.ballx, self.bally = np.random.randint(self.size), 4
            
        self.image[self.bally, self.ballx] = 1
        self.image[-5, self.pos - 2:self.pos + 3] = np.ones(5)
        
        for i in range(0, self.size):
            for j in range(0, self.size):
                self.image[i][j] = random.randint(2,5)
        
        return self.step(2)[0]


    def step(self, action):
        def left():
            if self.pos > 3:
                self.pos -= 2
        def right():
            if self.pos < 17:
                self.pos += 2
        def noop():
            pass
        {0: left, 1: right, 2: noop}[action]()

        
        self.image[self.bally, self.ballx] = 0
        self.ballx += self.vx
        self.bally += self.vy
        if self.ballx > self.size - 1:
            self.ballx -= 2 * (self.ballx - (self.size-1))
            self.vx *= -1
        elif self.ballx < 0:
            self.ballx += 2 * (0 - self.ballx)
            self.vx *= -1
        
        self.image[self.bally, self.ballx] = 1
        self.image[-5].fill(random.randint(2,5))
        self.image[-5, self.pos-2:self.pos+3] = np.ones(5)
    
        terminal = self.bally == self.size - 1 - 4
        reward = int(self.pos - 2 <= self.ballx <= self.pos + 2) if terminal else 0

        [self.state.append(imresize(self.image, (84, 84))) for _ in range(self.fps - len(self.state) + 1)]
        self.state = self.state[-self.fps:]

        return np.transpose(self.state, [1, 2, 0]), reward, terminal

    def get_num_actions(self):
        return 3

    def reset(self):
        return self.reset_random()

    def state_shape(self):
        return (self.fps,) + self.output_shape
    
    def show_state(self, i):
        plt.imshow(self.image)
        plt.imsave('image_'+str(i)+'.jpg', self.image)

def test():
    env = CatchEnv2()
    i = 0
    for ep in range(1):
        env.reset()
        env.show_state(i)
        
        state, reward, terminal = env.step(1) 
        while not terminal:
            env.show_state(i)
            state, reward, terminal = env.step(np.random.randint(0,2))
            i += 1
            #print(reward)

if __name__ == "main":
    test()
