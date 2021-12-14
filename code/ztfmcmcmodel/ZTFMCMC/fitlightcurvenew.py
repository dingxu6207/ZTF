# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:11:41 2021

@author: dingxu
"""

import numpy as np
import corner
import emcee
import matplotlib.pyplot as plt
import torch
import pickle,time
from MLPnet_reg import NET as NET 



####读入观测数据
#path = ''
#file = 'KIC 10389809.txt'
path = 'E:\\shunbianyuan\\data\\kepler\\KIC_name\\'
file = 'KIC 12305537.txt'
data = np.loadtxt(path+file)
phrase = data[:,0]
datay = data[:,1]-np.mean(data[:,1])
x = np.linspace(0,1,100) #x轴
noisy = np.interp(x,phrase,datay) #y轴
sigma=np.diff(noisy,2).std()/np.sqrt(6) #估计观测噪声值
#sigma=1

###########MCMC参数
nwalkers = 20
niter = 500
nburn=200 #保留最后多少点用于计算

init_dist = [(50.,90.),(2,9),(0,0.1),(0.9,1.0)] #初始范围,[incl,q,f,t2t1]
priors=init_dist.copy()
ndim = len(priors) #维度数



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #判断是否有GPU

def loadnet(): #load NETWORK
    netfile='dx_m_new.mod' #模型文件名
    
    inputCH=4#输入参数个数
    outputCH=100 #回归参数个数
    convNUM=500 #隐藏层卷积核个数，可修改
    layers=2#残差块数量，每块有两个卷积层，可修改
    
    model_state_dict = torch.load(netfile,map_location='cpu').state_dict()
    model = NET(inputCH,convNUM,layers,outputCH).to(device)
    model.load_state_dict(model_state_dict)
    model.eval() #为了防止BN层和dropout的随机性，直接用evaluation方式训练
    return(model)

def predict( allpara,model): #从参数产生光变
    npallpara = np.array(allpara) 
    npallpara = npallpara.astype('float32')
    Input=torch.from_numpy(npallpara).to(device) 
    output = model(Input) #训练！
    output = output.cpu().detach().numpy() #预测值
    return output

def rpars(init_dist):#在ndim 维度上，在初始的范围里面均匀撒ndim个点
    return [np.random.rand() * (i[1]-i[0]) + i[0] for i in init_dist] 


def lnprior(priors, values):#判断MCMC新的点是否在初始的区域里面
    
    lp = 0.
    for value, prior in zip(values, priors):
        if value >= prior[0] and value <= prior[1]:
            lp+=0
        else:
            lp+=-np.inf 
    return lp


def lnprob(z): #计算后仰概率
    
    lnp = lnprior(priors,z)#判断MCMC新的点是否在初始的区域里面

    if not np.isfinite(lnp):
            return -np.inf


    output=predict(z,model)

    lnp = -0.5*np.sum(np.log(2 * np.pi * sigma ** 2)+(output-noisy)**2/(sigma**2)) #计算似然函数
      
    return lnp


def run(init_dist, nwalkers, niter,nburn):
    
    ndim = len(init_dist)
    # Generate initial guesses for all parameters for all chains
    p0 = [rpars(init_dist) for i in range(nwalkers)] #均匀撒ndim*nwalkers点
 #   print(p0)

    # Generate the emcee sampler. Here the inputs provided include the 
    # lnprob function. With this setup, the first parameter
    # in the lnprob function is the output from the sampler (the paramter 
    # positions).
    sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob) #建立MCMC模型
    pos, prob, state = sampler.run_mcmc(p0, niter) # 撒点
    emcee_trace = sampler.chain[:, -nburn:, :].reshape(-1, ndim).T #保留最后nburn 个点做统计

    return emcee_trace 

model=loadnet() #加载神经网络

t1=time.time()
emcee_trace  = run(priors, nwalkers, niter,nburn) #run mcmc
print('time=',time.time()-t1) #MCMC运行时间


mu=(emcee_trace.mean(axis=1)) #参数均值
sigma=(emcee_trace.std(axis=1)) #参数误差
print('mu=',mu)
print('sigma=',sigma)


####################绘图
figure = corner.corner(emcee_trace.T,bins=100,labels=[r"$incl$", r"$q$", r"$f_0$", r"$t2t1$"],
                       label_kwargs={"fontsize": 15},show_titles=True, title_kwargs={"fontsize": 15}, color ='blue')
plt.savefig('corner.png')
#------------------------------------------------------------
#用输出值预测理论曲线
pre=predict(mu.reshape(1,-1),model)
plt.figure()
ax = plt.gca()
ax.plot(x,noisy,'.') #原始数据
ax.plot(x,pre.flatten(),'-r') #理论数据

ax.yaxis.set_ticks_position('left') #将y轴的位置设置在右边
ax.invert_yaxis() #y轴反向
