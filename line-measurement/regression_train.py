# Energy price non-linear regression
# solve for oil sales price (outcome)
# using 3 predictors of WTI Oil Price,
#   Henry Hub Price and MB Propane Spot Price
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data file from URL address
df = pd.read_csv("frame_test.csv")
#dataset = dataframe.values
#print(dataset)

# split into input (X) and output (Y) variables
#X = dataset[:,0:2]
#print(X.shape)
#Y = dataset[:,3]
#Y=pd.DataFrame(dataframe,columns=['dis'])
#X=dataframe.drop('dis',axis=1)

xm1 = np.array(df["index_of_curve"])  # WTI Oil Price
xm2 = np.array(df["max_angle_difference"])   # Henry Hub Gas Price
xm3 = np.array(df["point_dis"])  # MB Propane Spot Price
ym = np.array(df["dis"])  # oil sales price received (outcome)

# calculate y
def calc_y(x):
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    e=x[4]
    f=x[5]
    g=x[6]
    h=x[7]
    i=x[8]
    #y = a * xm1 + b  # linear regression
    y = (+a*xm1**2+b*xm1)+(b*xm2**2+c*xm2)+(f*(xm3**e)+d*xm3**2+e*xm3)+f
    return y

# define objective
def objective(x):
    # calculate y
    y = calc_y(x)
    # calculate objective
    obj = 0.0
    for i in range(len(ym)):
        obj = obj + ((y[i]-ym[i])/ym[i])**2    
    # return result
    return obj

# initial guesses
x0 = np.zeros(9)
x0[0] = 0.0 # a
x0[1] = 0.0 # b
x0[2] = 0.0 # c
x0[3] = 0.0 # d
x0[4]=0.0
x0[5]=0.0
x0[6]=0.0
x0[7]=0.0
x0[8]=0.0
# show initial objective
print('Initial Objective: ' + str(objective(x0)))

# optimize
# bounds on variables
my_bnds = (-100.0,100.0)
bnds = (my_bnds, my_bnds, my_bnds, my_bnds,my_bnds,my_bnds,my_bnds,my_bnds,my_bnds)
solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
x = solution.x
y = calc_y(x)

# show final objective
cObjective = 'Final Objective: ' + str(objective(x))
print(cObjective)

# print solution
print('Solution')

cA = 'a = ' + str(x[0])
print(cA)
cB = 'b = ' + str(x[1])
print(cB)
cC = 'c = ' + str(x[2])
print(cC)
cD = 'd = ' + str(x[3])
print(cD)
cE = 'e = ' + str(x[4])
print(cE)
cF = 'f = ' + str(x[5])
print(cF)
cF = 'g = ' + str(x[6])
print(cF)
cF = 'h = ' + str(x[7])
print(cF)
cF = 'i = ' + str(x[8])
print(cF)
cFormula = "Formula is : " + "\n" \
           + "A * WTI^B * HH^C * PROPANE^D"
cLegend = cFormula + "\n" + cA + "\n" + cB + "\n" \
           + cC + "\n" + cD+ cE+ cF + "\n" + cObjective

#ym measured outcome
#y  predicted outcome

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(ym,y)
r2 = r_value**2
cR2 = "R^2 correlation = " + str(r_value**2)
print(cR2)

# plot solution
plt.figure(1)
plt.title('Actual (YM) versus Predicted (Y) Outcomes For Non-Linear Regression')
plt.plot(ym,y,'o')
plt.xlabel('Measured Outcome (YM)')
plt.ylabel('Predicted Outcome (Y)')
#plt.legend([cLegend])
plt.grid(True)
plt.show()