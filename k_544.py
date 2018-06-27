# -*- coding:utf-8 -*-

import numpy as np
from matplotlib import pyplot

# 初期条件
def f(x, u00):
    
    ux0 = np.exp(-100*(x-0.3)**2)
    ux0[0] = u00

    return ux0

# 空間微分
def dudx(x, dx):

    y = np.zeros_like(x) # 初期化

    for i in range(1,len(x)-1):
        y[i] = (x[i+1]-x[i-1])/(2*dx)

    y[-1] = (x[-1]-x[-2])/dx

    return y

# 時間微分(Runge-Kutta)
def R_K(unow, dx, dt):

    # 初期化
    du = 0
    S = np.zeros_like(unow)

    # 係数ak,bk
    a = np.array([0, 1/2, 1/2, 1])
    b = np.array([1/6, 2/6, 2/6, 1/6])

    # Runge-Kutta
    for k in range(4):
        u_ = unow + a[k] * du
        du = -1 * dudx(u_, dx) * dt
        S = S + b[k] * du

    unext = unow + S

    return unext


def main():

    # メッシュ
    dx = 10**(-2)
    dt = 10**(-3)

    # x,tベクトル定義
    x = np.arange(0,1,dx)
    t = np.arange(0,1,dt)

    # 境界条件
    u0t = 0

    # 初期条件
    ux0 = f(x, u0t)

    # (t,x)行列を定義、初期化
    u = np.zeros((len(t), len(x)))

    # (0,x)=(初期条件)
    u[0] = ux0                     

    # Runge-Kutta
    for i in range(1, len(t)):
        u[i] = R_K(u[i-1], dx, dt)

    # t=100*dt 毎に表示
    for j in range(len(t)):
        if (j%100)==0:
            pyplot.plot(x, u[j])

    pyplot.show()


if __name__ == '__main__':
    main()
