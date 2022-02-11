# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 09:31:25 2022

@author: jkf
"""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pyro,time
import pyro.contrib.gp as gp
import pyro.distributions as dist

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.0')
#pyro.set_rng_seed(0)

def plot(plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
         model=None, kernel=None, n_test=100):

    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        plt.plot(X.numpy(), y.numpy(), 'kx')
    if plot_predictions:
        Xtest = torch.linspace(mi, mx, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        #print(sd)
        plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                         (mean - 2.0 * sd).numpy(),
                         (mean + 2.0 * sd).numpy(),
                         color='C0', alpha=0.3)
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(mi, mx, n_test)  # test inputs
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    plt.xlim(mi, mx)
    


plt.close('all')

# dat=torch.tensor(np.loadtxt('magtemprature.txt').astype('float32'))
# X=dat[:,0]-dat[:,1]
# y=1/dat[:,2]*30000
#y/=2000
dat=torch.tensor(np.loadtxt('ztf.txt').astype('float32'))
#dat=dat[::10]
X=dat[:,0]
y=dat[:,1]
mi=X.min()
mx=X.max()
y=y-y.mean()

N=dat.shape[0]
#plot(plot_observed_data=True)

# initialize the inducing inputs
Xu = (torch.arange(10.) / 10.0+mi)*(mx-mi)

# initialize the kernel and model
pyro.clear_param_store()
t1=time.time()
kernel = gp.kernels.RBF(input_dim=1)
noise=np.diff(y,2).std()/np.sqrt(6)
noise=noise.astype('float32')
# we increase the jitter for better numerical stability
sgpr = gp.models.SparseGPRegression(X, y, kernel, Xu=Xu, noise=torch.tensor(noise),jitter=1.0e-6)
#sgpr = gp.models.SparseGPRegression(X, y, kernel, Xu=Xu, jitter=1.0e-4)
#sgpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(noise)) 
# the way we setup inference is similar to above
optimizer = torch.optim.Adam(sgpr.parameters(), lr=0.005)
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
num_steps = 2000 if not smoke_test else 2
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(sgpr.model, sgpr.guide)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if(i%100==0):
            print(i,loss.item())
print(time.time()-t1)            
plt.figure()            
plt.plot(losses);

#print("inducing points:\n{}".format(sgpr.Xu.data.numpy()))
# and plot the predictions from the sparse GP
plot(model=sgpr, plot_observed_data=True, plot_predictions=True)
