import numpy as np
import pandas as pd
import math
import time
import argparse
from itertools import count
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy import stats

class growth_model():
    def __init__(self):
        self.mI,self.mR, self.LuxI, self.LuxR, self.AHLin, self.AR, self.AR2=[4],[12],[800],[92],[13],[77],[555]
        self.mLL37, self.LL37, self.mtetR, self.tetR, self.mRFP, self.RFP=[50],[820],[26],[950],[19],[120]
        self.STATH, self.BMP2=[0.2],[0.0042]
        self.Ecoli, self.bacteria, self.E, self.b=[0.3],[float(input())],[0],[0]
        self.tans = 1
        self.time = [0]
        self.eat = 0
        self.total = [0]
    def __call__(self, delta_t):
        self.bacteria[self.tans - 1] = 0.6
        delta_t += 240
        self.reward = 0
        poisson_x_axis = np.arange(0,delta_t)
        poisson_y_axis = stats.poisson.cdf(poisson_x_axis,360)
        a = 0
        total = 0
        for i in range(self.tans-1, self.tans+delta_t-1):
            a = int(a+1)
            b = poisson_y_axis[a]
            c = random.uniform(0,1)
            if b >= c:
                self.eat = 120
                a = 0
            if self.eat != 0:
                self.bacteria[i] = self.bacteria[i] + 0.0001
                self.eat = self.eat - 1
            self.time.append(i)
            if i >= self.tans-1 and i <= self.tans+59:
                self.Ecoli[i] = self.Ecoli[i] + 0.001
            self.mI1=self.mI[i]+1.0017 - 0.247 *self.mI[i]
            self.mI.append(self.mI1)
            self.mR1=self.mR[i]+ 3.0050 - 0.247*self.mR[i]
            self.mR.append(self.mR1)
            self.LuxI1=self.LuxI[i]+5.28*self.mI[i]- 0.027*self.LuxI[i]
            self.LuxI.append(self.LuxI1)
            self.LuxR1=self.LuxR[i]+4.08*self.mR[i]-0.2*self.LuxR[i] -0.06624*self.LuxR[i]*self.AHLin[i]+ 0.6624*self.AR[i]
            if self.LuxR1<0 or self.LuxR1>190:
                self.LuxR.append(self.LuxR[i])
            else:
                self.LuxR.append(self.LuxR1)

            self.AHLin1=self.AHLin[i]+0.04*self.LuxI[i]-0.057*self.AHLin[i]-0.06624*self.LuxR[i]*self.AHLin[i]+ 0.6624*self.AR[i]
            if self.AHLin1<0 or self.AHLin1>10:
                self.AHLin.append( self.AHLin[i])
            else:
                self.AHLin.append(self.AHLin1)

            self.AR1=self.AR[i]+0.06624*self.LuxR[i]*self.AHLin[i] - 0.6624*self.AR[i] - 0.156*self.AR[i] -0.03312*((self.AR[i])**2)+ 0.3312*self.AR2[i]
            if self.AR1<0 or self.AR1>65:
                self.AR.append( self.AR[i])
            else:
                self.AR.append(self.AR1)
        
            self.AR21=self.AR2[i]+0.03312*((self.AR[i])**2) - 0.3312*self.AR2[i] - 0.017*self.AR2[i]
            if self.AR21<0 or self.AR21>380:
                self.AR2.append(self.AR2[i])
            else:
                self.AR2.append(self.AR21)
            
            self.mLL371=self.mLL37[i]+52.8*((200+0.01*self.AR2[i])/(200+self.AR2[i]))-0.288*self.mLL37[i]
            if self.mLL371<0 or self.mLL371>180:
                self.mLL37.append( self.mLL37[i])
            else:
                self.mLL37.append(self.mLL371)
        
            self.LL371=self.LL37[i]+3.5*self.mLL37[i]*self.Ecoli[i]*1.07*(10**12)*2.534*(10**(-13)) - 0.011*self.LL37[i]-4*(10**(-5))*3.5*(10**(-4))*(self.b[i]+self.E[i])*self.LL37[i]
            if self.LL371<0 or self.LL371>1500:
                self.LL37.append(self.LL37[i])
            else:
                self.LL37.append(self.LL371)

            self.mtetR1=self.mtetR[i]+52.8*((200+0.01*self.AR2[i])/(200+self.AR[i]))-0.54*self.mtetR[i]
            if self.mtetR1<0 or self.mtetR1>100:
                self.mtetR.append( self.mtetR[i])
            else:
                self.mtetR.append( self.mtetR1)
        
            self.tetR1=self.tetR[i]+4.9275*self.mtetR[i]-0.1386*self.tetR[i]
            if self.tetR1<0 or self.tetR1>3300:
                self.tetR.append(self.tetR[i])
            else:
                self.tetR.append(self.tetR1)

            self.mRFP1=self.mRFP[i]+52.8*((200+0.01*self.AR2[i])/(200+self.AR2[i]))-0.8*self.mRFP[i]
            if self.mRFP1<0 or self.mRFP1>70:
                self.mRFP.append(self.mRFP[i])
            else:
                self.mRFP.append(self.mRFP1)
        
            self.RFP1=self.RFP[i]+4.5333*self.mRFP[i]*self.Ecoli[i]*1.07*(10**12)*2.534*(10**(-13)) - 0.04*self.RFP[i]
            if self.RFP1<0 or self.RFP1>400:
                self.RFP.append(self.RFP[i])
            else:
                self.RFP.append(self.RFP1)

            self.STATH1 =self.STATH[i]+1.835*(0.002+(1-0.002)/(1+(self.tetR[i]/3.946)**3))*self.Ecoli[i]*1.07*(10**12)*2.534*(10**(-13)) -0.0000248*self.STATH[i]
            if self.STATH1<0 or self.STATH1>0.04:
                self.STATH.append( self.STATH[i])
            else:
                self.STATH.append(self.STATH1)

            self.BMP21=self.BMP2[i]+1.835*(0.002+(1-0.002)/(1+(self.tetR[i]/3.946)**3))*self.Ecoli[i]*1.07*(10**12)*2.534*(10**(-13))-0.05*self.BMP2[i]
            if self.BMP21<0 or self.BMP21>0.019:
                self.BMP2.append( self.BMP2[i])

            else:
                self.BMP2.append(self.BMP21)

            if i <= 1000:
                self.Ecoli1 = self.Ecoli[i] + 0.04*self.Ecoli[i] * (1-(self.Ecoli[i]/1.5)) - self.LL37[i] * 0.00004*self.Ecoli[i]
            else:
                self.Ecoli1 = self.Ecoli[i] - self.LL37[i] * 0.00004 * self.Ecoli[i]
                
            if self.Ecoli1 < 0 :
                self.Ecoli.append(self.Ecoli[i])
            else:
                self.Ecoli.append(self.Ecoli1)
            self.bacteria1 = self.bacteria[i] + 0.13 * self.bacteria[i] * (1-(self.bacteria[i]/0.7)) - self.LL37[i] * 0.00004 * self.bacteria[i]
            
            if self.bacteria1 < 0 or self.bacteria1 > 2.5:
                self.bacteria.append(self.bacteria[i])
            else:
                self.bacteria.append(self.bacteria1)
            
            self.E1=self.E[i]+self.LL37[i]*0.00004*self.Ecoli[i]
            if self.E1<0 :
                self.E.append(self.E[i])
            else:
                self.E.append(self.E1)

            self.b1=self.b[i]+self.LL37[i]*0.00004*self.bacteria[i]
            if self.b1<0 :
                self.b.append(self.b[i])
            else:
                self.b.append(self.b1)
            self.reward = self.reward + self.Ecoli[i+1] + self.bacteria[i+1]
        self.eat = 0
        self.reward =(delta_t-240)*100/self.reward
        self.number = self.Ecoli1 + self.bacteria1
        self.tans = self.tans + delta_t
        return self.number, delta_t, self.reward, self.Ecoli1 ,self.bacteria1


parser = argparse.ArgumentParser(description='iGEM NCTU_FORMOSA 2021 Dentalbone utilizing optimization reinforcement learning model')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.6)
        self.rewards = []

    def forward(self, bacteria, delta_t):
        x = torch.cat((bacteria, delta_t), dim = 0)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.dropout(x)
        x = self.fc3(x)
        action = delta_t*torch.sigmoid(x)
        return action

policy = Policy()
my_growth_model = growth_model()
optimizer = optim.Adam(policy.parameters(), lr=1)
eps = np.finfo(np.float32).eps.item()

def change_action(bacteria, delta_t):
    bacteria = torch.from_numpy(np.array(bacteria)).float().unsqueeze(0)
    delta_t = torch.from_numpy(np.array(delta_t)).float().unsqueeze(0)
    action = policy.forward(bacteria, delta_t)
    return action


def finish_episode(R,reward):
    R.append(args.gamma*reward)
    policy_loss = []
    policy_loss.append(-1*math.log(reward))
    optimizer.zero_grad() 
    policy_loss = Variable(torch.tensor(policy_loss), requires_grad = True)
    policy_loss.backward()
    optimizer.step()

def main():
    R = []
    bacteria_list = []
    Pgingivalis_list = []
    Ecoli_list = []
    bacteria  = np.array(0.4, dtype = float)
    delta_t = np.array(360, dtype = float)
    running_reward_list = []
    running_reward = 0
    total_reward_list = []
    total_reward = 0
    delta_t_list = []
    for i_episode in range(100):
        ep_reward = 0
        action = change_action(bacteria, delta_t)
        bacteria, delta_t, reward, Ecoli, Pgingivalis  = my_growth_model(int(action))
        delta_t_list.append(delta_t)
        policy.rewards.append(reward)
        ep_reward += reward
        total_reward += reward
        total_reward_list.append(total_reward)
        bacteria_list.append(bacteria)
        Pgingivalis_list.append(Pgingivalis)
        Ecoli_list.append(Ecoli)
        
        if i_episode == 0:
            running_reward = ep_reward
        else:
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        running_reward_list.append(running_reward)
        finish_episode(R,reward)
    
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAveraqge reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
if __name__ == '__main__':
    main()