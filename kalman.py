# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

# by Andrew D. Straw

import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import datasets, linear_model
import statsmodels.api as sm


#import the data
df_train = pandas.read_csv("intensities2.csv")
df_train['normalized'] = df_train['emission1'] / df_train['area']

#split data collected, test filter against 67%
target_dc = 42
#df_predict = df_train[df_train.dutycycle==target_dc]
#df_predict['normalized'] = df_predict['emission1']/df_train['area']
#df_train = df_train[df_train.dutycycle!=target_dc]
#df_train['normalized'] = df_train['emission1']/df_train['area']
df_predict = pandas.read_csv('baseline-42.csv').iloc[5:400]
df_predict['normalized'] = df_predict['emission1'] / df_predict['area']
means = np.zeros(len(df_predict))
for i in range(len(means)):
    means[i] = df_predict['normalized'][:i].mean()


df_led = pandas.read_csv("LED.csv")
df_led = df_led[df_led > 8]
camera_variance = df_led['emission1'].var()


true_value = df_predict['normalized'].mean()

plt.rcParams['figure.figsize'] = (10, 8)

#determine the model
X = sm.add_constant(df_train["dutycycle"].values.reshape(-1,1))
ols = sm.OLS(df_train["normalized"], X)
ols_result = ols.fit()

#regr = linear_model.LinearRegression()
#regr.fit(df_train["dutycycle"].values.reshape(-1,1), df_train["emission1"])
fit_b = ols_result.params.const
fit_m = ols_result.params.x1
model_variance = ols_result.bse.const **2

# intial parameters
n_iter = 195
sz = (n_iter, 2) # size of array
x = true_value # truth value (typo in example at top of p. 13 calls this z)
#z = np.random.normal(x,0.1,size=n_iter) # observations (normal about x, sigma=0.1)
z = df_predict["normalized"].values

#Q = 1e-5 # process variance
#Q = np.array(model_variance, model_variance)
Q = np.eye(2) * model_variance * 0.001
Q[0,0] *= 0.1

# allocate space for arrays
xhat=np.zeros(sz)      # a posteri estimate of x
P=np.zeros((n_iter, 2, 2))         # a posteri error estimate
xhatminus=np.zeros(sz) # a priori estimate of x
Pminus=np.zeros((n_iter, 2, 2))    # a priori error estimate
K=np.zeros((n_iter, 2, 2))         # gain or blending factor

#R = 0.1**2 # estimate of measurement variance, change to see effect
#R = np.array(camera_variance, camera_variance)
R = np.eye(2) * camera_variance * 1
R[0,0] *= 10

#x[0] = intercept
#x[1] = actual state ????

# intial guesses
#xhat[0] = 0.0
# use our linear model...
fit_b = df_predict['normalized'].iloc[0] - target_dc * fit_m
xhat[0,0] = fit_b
xhat[0,1] = target_dc * fit_m + fit_b
#what
#P[0] = np.array(1.0,1.0)
P[0] = np.eye(2) 
#P[0,0] =

for k in range(1,n_iter):
    # time update
    #xhatminus[k] = xhat[k-1]
    # Don't make any new predictions for the intercept, use the measurement for that instead
    xhatminus[k,0] = xhat[k-1, 0]
    # Use model to make a new prediction for the value
    xhatminus[k,1] = xhatminus[k, 0] + fit_m * target_dc
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k,0] = xhatminus[k,0]+K[k,0,0]*(z[k] - fit_m * target_dc - xhatminus[k,0]) #* 0.997**k
    #xhat[k,0] = xhatminus[k,0]#+K[k,0,0]*(z[k] - fit_m * target_dc - xhatminus[k,0])# * 0.95**k
    xhat[k,1] = xhatminus[k,1]+K[k,1,1]*(z[k] -xhatminus[k,1])
    P[k] = (1-K[k])*Pminus[k]

plt.figure()
plt.plot(z,'k+',label='noisy measurements')
plt.plot(xhat[:,1],'b-',label='a posteri estimate')
plt.plot(means, label='simple mean')
plt.axhline(x,color='g',label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Measurement number')
plt.ylabel('Average pixel intensity'
)
plt.savefig('kalman.png')

plt.figure()
valid_iter = range(1,n_iter) # Pminus not valid at step 0
plt.plot(valid_iter,Pminus[valid_iter,1,1],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(Voltage)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()
