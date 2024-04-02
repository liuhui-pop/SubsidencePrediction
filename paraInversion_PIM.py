import ch
ch.set_ch()
import numpy as np
import math
from scipy import integrate
from scipy.optimize import curve_fit
import csv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 坐标转换
def coordTransform(x, y, p1):
    for i in range(len(x)):
        xx = x - [p1[0], p1[1]]
        yy = np.cos(p1[2]), np.sin(p1[2])
        -np.sin(p1[2]), np.cos(p1[2])*xx


# 计算沉陷值的被积函数，针对u积分，x每次为定值
def funcNorm(u, x, r):
    return np.exp(-np.pi*pow(u - x, 2)/pow(r, 2))


# xy 是坐标向量，返回向量value-------------可以拟合反演参数，但是结果不对
def funcPIM(xy, M, q, r):
    n = xy.shape[0]
    f = np.zeros((n, 2))
    for i in range(n):
        f[i, 1] = integrate.romberg(funcNorm, 0, 500, args=(xy[i,1], r))
        f[i, 0] = integrate.romberg(funcNorm, 0, 600, args=(xy[i,0], r))
    # value = -M * q * np.cos(alpha * np.pi / 180) * 1000 * f[:, 1] * f[:, 0] / pow(r, 2)
    value = -M * q * 1000 * f[:, 1] * f[:, 0] / pow(r, 2)
    return value


def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    pdf = (pdf - min(pdf))/(max(pdf)-min(pdf)) * -3000
    return pdf


def generatData(n):
    result = np.random.randint(0, 760, 900) # 最小值，最大值，size
    x = np.arange(min(result), max(result), 1)
    y = normfun(x, result.mean(), result.std())
    x = x[:n] # n=750
    y = y[:n]
    return y


# 写文件csv
def writecsv(x, y, value, outfilename):
    dir = 'D:\\0. Papers\\myCode\\mineSubsidence'
    file_name = os.path.join(dir, outfilename + '.csv')
    csvfile = open(file_name, 'wt', newline='', encoding="UTF8")
    writer=csv.writer(csvfile, delimiter=",")
    header=['lat','lon','value']
    writer.writerow(header)
    writer.writerows(zip(x, y, value))
    csvfile.close()


if __name__ == '__main__':
    L0 = 1900 # 工作面走向长度m   500
    L1 = 300 # 工作面倾向长度m   600
    alpha = 1 # 煤层倾角°
    H0 = 160 # 平均开采深度m   220
    M = 5 # 开采厚度m   6.1

    # 预计参数
    q = 0.6 # 下沉系数  0.648
    tanB = 2.87 # 主要影响角正切
    b = 0.51 # 水平移动系数
    r = H0 / tanB # 主要影响半径
    theta = 76.4  # 开采影响传播角

    # x方向为倾向长，y方向为走向长
    res_x = 20 # x方向一个pixel的尺度，单位m
    res_y = 20 # y方向一个pixel的尺度
    columns = round(L1 / res_x) # x方向pixel的个数
    rows = round(L0 / res_y) # y方向pixel的个数

    # 极值计算
    Wmax = M * q * np.cos(alpha * np.pi / 180) * 1000 # 最大下沉值 /mm
    imax = Wmax / r # 最大倾斜值
    Kmax = 1.52 * Wmax / pow(r, 2) # 最大曲率值
    Umax = b * Wmax # 最大水平移动值
    Epsilonmax = 1.52 * b * Wmax / r # 最大水平变形值
    print('最大下沉值:', Wmax, '\n最大倾斜值:', imax, '\n最大曲率值:', Kmax, '\n最大水平移动值:', Umax, '\n最大水平变形值:', Epsilonmax)

    # 计算任意点下沉值

    Y, X = np.meshgrid(range(rows),range(columns))
    lat = np.squeeze(X).reshape(columns*rows, )
    lon = np.squeeze(Y).reshape(columns*rows, )
    # subsidence = []
    # for i in range(columns):
    #     for j in range(rows):
    #         ptSub = funcPIM(i*res_x, j*res_y, M, q, alpha, r, L0, L1)
    #         subsidence.append(ptSub)
    # # writecsv(lat, lon, subsidence, 'subSidence')
    # print(max(subsidence))
    # print(min(subsidence))

    # 计算任意点下沉值
    xy = np.zeros((columns*rows,2))
    xy[:,0] = lat*res_x
    xy[:,1] = lon*res_y
    data = np.array([M, q, r])  #  6, 0.65, 220/2.87 = 76.65
    # subsidence = funcPIM(xy, M, q, r)
    subsidence = funcPIM(xy, M, q, r)
    print(subsidence.shape)
    a = 0
    # 随机生成实测值序列
    # real = generatData(xy.shape[0])
    # plt.plot(np.arange(0,750), real)
    # plt.show()
    # print(min(real),max(real),real.shape)

    # 拟合预计参数
    popt, pcov = curve_fit(funcPIM, xy, subsidence)
    print(popt)

    subsidence_fit = funcPIM(xy, popt[0], popt[1], popt[2]) # 拟合值

    plt.plot(np.arange(0,columns*rows), subsidence, 'r', label = 'real values')
    plt.plot(np.arange(0,columns*rows), subsidence_fit,'b', label = 'fit values')
    plt.legend()
    plt.show()

    # 画图---概率积分法计算结果
    Z = np.reshape(np.array(subsidence), (columns, rows))
    print(Z.shape)
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    plt.title("矿区沉陷预计",fontsize=20)
    ax.plot_surface(X, Y, Z, cmap = plt.get_cmap('rainbow'))
    # a = plt.contourf(X, Y, Z, cmap = plt.get_cmap('rainbow'))
    # b = plt.contour(X, Y, Z, colors='black', linewidths=1, linestyles='solid')

    # 三个面上的等高线投影
    cz = ax.contour(X, Y, Z, zdir='z', offset=-5000, cmap=plt.get_cmap('rainbow'))
    # cx = ax.contour(X, Y, Z, zdir='x', offset=5, cmap=plt.get_cmap('rainbow'))
    # cy = ax.contour(X, Y, Z, zdir='y', offset=5, cmap=plt.get_cmap('rainbow'))
    # plt.colorbar(a, ticks=[0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12])
    # plt.colorbar(a)
    # plt.clabel(b, inline=True, fontsize=10)
    ax.set_xlim(0,L1/res_x)
    ax.set_ylim(0,L0/res_y)
    ax.set_zlim(-5000,0)
    ax.set_xlabel('工作面倾向/m',fontsize=20)
    ax.set_ylabel('工作面走向/m',fontsize=20)
    ax.set_zlabel('沉陷值/mm',fontsize=20)
    # ax.set_xticklabels(np.arange(0,220,30).tolist())
    # ax.set_yticklabels(np.arange(0,200,33).tolist())
    # ax.set_zlim(0,0.15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 画图---拟合结果
    Z = np.reshape(np.array(subsidence_fit), (columns, rows))
    print(Z.shape)
    fig2 = plt.figure()
    ax = Axes3D(fig2)
    plt.title("矿区沉陷预计-拟合结果",fontsize=20)
    ax.plot_surface(X, Y, Z, cmap = plt.get_cmap('rainbow'))
    # a = plt.contourf(X, Y, Z, cmap = plt.get_cmap('rainbow'))
    # b = plt.contour(X, Y, Z, colors='black', linewidths=1, linestyles='solid')

    # 三个面上的等高线投影
    cz = ax.contour(X, Y, Z, zdir='z', offset=-5000, cmap=plt.get_cmap('rainbow'))
    # cx = ax.contour(X, Y, Z, zdir='x', offset=5, cmap=plt.get_cmap('rainbow'))
    # cy = ax.contour(X, Y, Z, zdir='y', offset=5, cmap=plt.get_cmap('rainbow'))
    # plt.colorbar(a, ticks=[0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12])
    # plt.colorbar(a)
    # plt.clabel(b, inline=True, fontsize=10)
    ax.set_xlim(0,L1/res_x)
    ax.set_ylim(0,L0/res_y)
    ax.set_zlim(-5000,0)
    ax.set_xlabel('工作面倾向/m',fontsize=20)
    ax.set_ylabel('工作面走向/m',fontsize=20)
    ax.set_zlabel('沉陷值/mm',fontsize=20)
    # ax.set_xticklabels(np.arange(0,220,30).tolist())
    # ax.set_yticklabels(np.arange(0,200,33).tolist())
    # ax.set_zlim(0,0.15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()
