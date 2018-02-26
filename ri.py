import os
import random
import time
from random import randint, random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import (NumericProperty, ObjectProperty,ReferenceListProperty)
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.vector import Vector
from torch.autograd import Variable


class Network(nn.Module):

    def __init__(self, input_size , nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1= nn.Linear(input_size , 30)
        self.fc2= nn.Linear(30, nb_action)

    def forward(self,state):
        x= F.relu(self.fc1(state))
        q_values= self.fc2(x)
        return q_values

    class ReplayMemory(object):

        def __init__(self,capacity):
            self.capacity= capacity
            self.memory =[]

        def push(self, event):
            self.memory.append(event)
            if len(self.memory)>self.capacity:
                del self.memory[0]

        def sample(self , batch_size):
            samples=zip(*random.sample(self.memory , batch_size))
            return map(lambda x: Variable(torch.cat(x, 0)), samples)

class Dqn():
    def __init__(self,input_size,nb_action,gamma):
        self.gamma=gamma
        self.reward_window= []
        self.model= Network(input_size, nb_action)
        self.memory = Network.ReplayMemory(100000)
        self.optimizer= optim.Adam(self.model.parameters(),lr=0.001)
        self.last_state=torch.Tensor(input_size).unsqueeze(0)
        self.last_action= 0
        self.last_reward= 0

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state , volatile= True))*7)
        action = probs.multinomial()
        return action.data(0,0)

    def learn(self, batch_state , batch_next_state , batch_reward , batch_action):
        outputs= self.model(batch_state).gather(1, batch_action).unsqueeze(1).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables= True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state , new_state , torch.LongTensor([int(self.last_action)]),torch.tensor([self.last_reward])))
        action= self.select_action(new_state)
        if len(self.memory.memory)>100:
            batch_state , batch_next_state , batch_reward , batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward , batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window)>1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        torch.save({'state_dict' : self.model.state_dict(), 'optimizer':self.optimizer.state_dict , } , 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print ("=> loading checkpoint ...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")

Config.set('input', 'mouse' , 'mouse,multitouch_on_demand')

last_x = 0
last_y = 0
n_points = 0
length = 0

brain = Dqn(5, 3, 0.9)
action2rotation = [0,20,-20]
last_reward = 0
scores = []

first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur, largeur))
    goal_x = 20
    goal_y = largeur-20
    first_upadate = False


last_distance = 0

class Car(Widget):

     angle = NumericProperty(0)
     rotation = NumericProperty(0)
     velocity_x= NumericProperty(0)
     velocity_y= NumericProperty(0)
     velocity= ReferenceListProperty(velocity_x , velocity_y)
     sensor1_x= NumericProperty(0)
     sensor1_y=NumericProperty(0)
     sensor1 = ReferenceListProperty(sensor1_x , sensor1_y )
     sensor2_x =NumericProperty(0)
     sensor2_y =NumericProperty(0)
     sensor2 =ReferenceListProperty(sensor2_x , sensor2_y)
     sensor3_x =NumericProperty(0)
     sensor3_y = NumericProperty(0)
     sensor3= ReferenceListProperty(sensor3_x , sensor3_y)
     signal1 = NumericProperty(0)
     signal2= NumericProperty(0)
     signal3 = NumericProperty(0)

     def move(self, rotation):
            self.pos= Vector(*self.velocity) + self.pos
            self.rotation = rotation
            self.sensor1 = Vector(10, 0).rotate(self.angle) + self.pos
            self.sensor2 = Vector(30,0 ).rotate(self.angle+30%360) +self.pos
            self.sensor3 = Vector(30,0 ).rotate(self.angle+30%360) +self.pos
            self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int (self.sensor1_x)+10,int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400
            self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int (self.sensor2_x)+10,int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400
            self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int (self.sensor3_x)+10,int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400
            if self.sensor1_x>largeur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
                self.signal1= 1.
            if self.sensor2_x>largeur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
                self.signal2= 1.
            if self.sensor3_x>largeur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
                self.signal3= 1.


class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

def serve_car(self):
    car= ObjectProperty(None)
    ball1= ObjectProperty(None)
    ball2= ObjectProperty(None)
    ball3 = ObjectProperty(None)
    
    def serve_Car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6,0)
        
    def update(self, dt):
        
        global brain 
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        
        longueur = self.width 
        largeur = self.height 
        if first_update:
            init()
            
        xx= goal_x- self.car.x
        yy= goal_y- self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1 , self.car.signal2 , self.car.signal3 , orientation , -orientation ]
        action= brain.update(last_reward , last_signal ) 
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y -goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        
        if sand[int(self.car.x),int(self.car.y)]>0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward= -1
        else:
            self.car.velocity = Vector(6,0).rotate(self.car.angle)
            last_Reward=-0.2
            if distance < last_distance :
                last_reward =0.1
                
        if self.car.x <10:
            self.car.x=10 
            last_reward=-1
        if self.car.x > self.width-10:
            self.car.x= self.width-10
            last_reward=-1
        if self.car.y <10:
            self.car.y= 10
            lsat_reward=-1
        if self.car.y >self.height-10:
            self.car.y=self.height-10
            lst_reward = -1
            
        if distance< 100:
            goal_x =self.width-goal_x
            goal_y =self.height-goal_y
        last_distance=distance 


class MyPaintWidget(Widget):

    def on_touch_down(self,touch):
        global length, n_points , last_x , last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d=10.
            touch.ud['line']=Line(points =(touch.x,touch.y) , width =10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points =0
            length= 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x,lst_y
        if touch.button =='left':
            touch.ud['line'].points +=[touch.x,touch.y]
            x= int(touch.x)
            y= int(touch.y)
            length +=np.sqrt(max((x-last_x)**2 + (y-last_y)**2,2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20*density+1)
            sand[int(touch.x) - 10 : int (touch.x) +10 , int(touch.y)-10 : int(touch.y)+10]
            last_x=x
            last_y=y


class CarApp(App):

    def build(self):
        parent= Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update , 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn =  Button(text ='clear')
        savebtn = Button(text='save' , pos =(parent.width,0))
        loadbtn = Button(text= 'load', pos= (2*parent.width ,0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self,obj):
        print("loading last saved brain...")
        brain.load()

if __name__ == '__main__':
    CarApp().run()
