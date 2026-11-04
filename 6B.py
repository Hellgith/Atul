import numpy as np
import random

locations = [(0,0), (0,4), (4,0), (4,4)]

def encode(r,c,p,d):
    return ((r*5+c)*5+p)*4+d

def decode(s):
    d=s%4; s//=4
    p=s%5; s//=5
    c=s%5; r=s//5
    return r,c,p,d

def reset():
    r,c=random.randrange(5),random.randrange(5)
    p,d=random.sample(range(4),2)
    return encode(r,c,p,d)

def move(s,a):
    r,c,p,d=decode(s)
    reward=-1; done=False

    if a==0: r=min(4,r+1)
    elif a==1: r=max(0,r-1)
    elif a==2: c=min(4,c+1)
    elif a==3: c=max(0,c-1)
    elif a==4:
        if p<4 and (r,c)==locations[p]: p=4; reward=10
        else: reward=-10
    elif a==5:
        if p==4 and (r,c)==locations[d]: p=d; reward=20; done=True
        else: reward=-10

    return encode(r,c,p,d),reward,done

Q=np.zeros((500,6))
alpha,gamma,epsilon=0.8,0.95,1.0

print("Training started...")

for ep in range(10000):
    state=reset()
    for t in range(200):
        action=random.randrange(6) if random.random()<epsilon else np.argmax(Q[state])
        ns,reward,done=move(state,action)
        Q[state,action]+=alpha*(reward+gamma*np.max(Q[ns])-Q[state,action])
        state=ns
        if done: break
    epsilon=max(0.01,epsilon*0.995)

print("Training completed!\nTesting...")

for i in range(10):
    state=reset()
    total_reward=0
    for t in range(200):
        state,reward,done=move(state,np.argmax(Q[state]))
        total_reward+=reward
        if done: break
    print(f"Episode {i+1}: Steps={t+1}, Reward={total_reward}")

print("\nQ-Learning completed successfully!")