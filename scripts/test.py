### --- Import and define ---
import numpy as np
import time
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue

def MSFT(queue, xx, yy, zz, RR, ind_img=None):
    if ind_img is not None:
        buff = np.zeros((len(ind_img), len(ind_img)), dtype=complex)
        for ii in range(len(zz)):
            rho = xx*xx + yy*yy <= RR*RR - zz[ii]*zz[ii]
            if sum(sum(rho)) == 0:
                continue
            buff2 = np.fft.fftshift(np.fft.fft2(rho))
            buff += buff2[ind_img[0]:ind_img[len(ind_img)-1]+1,
                           ind_img[0]:ind_img[len(ind_img)-1]+1]
        queue.put(buff)
    else:
        w,h = xx.shape
        buff = np.zeros((w, h), dtype=complex)
        for ii in range(len(zz)):
            rho = xx*xx + yy*yy <= RR*RR - zz[ii]*zz[ii]
            if sum(sum(rho)) == 0:
                continue
            buff += np.fft.fft2(rho)
        queue.put(np.fft.fftshift(buff))
    
### --- input parameters ---
R = 70.
N = 1024
xmax = 1000.0
Wavelength = 2.2 # [A]
process_num = 4


### --- Process ---
if __name__=="__main__":
    x0 = 4.5
    pnts = [int(N/process_num)*ii for ii in range(process_num+1)]
    sprange = np.linspace(-xmax, xmax, N)
    dx = sprange[1] - sprange[0]
    
    x,y = np.meshgrid(sprange, sprange)
    
    freq = np.linspace(-0.5/dx, 0.5/dx, N)
    q = 2*np.pi*freq
    k = 2*np.pi/Wavelength # Wave number [1/A]
    ind_ink = abs(q) <= k
    q = q[ind_ink]
    ind_img = np.where(ind_ink > 0)[0]
    
    b1 = np.zeros((len(ind_img), len(ind_img)), dtype=complex)
    #b1 = np.zeros((N, N), dtype=complex)
    queue = Queue()
    jobs = []
    for ii in range(process_num):
        jobs.append(
            Process(target=MSFT, args=(queue, x, y, sprange[pnts[ii]:pnts[ii+1]-1], R, ind_img)))

    st = time.time()
    print("start")
    for job in jobs:
        job.start()
    for job in jobs:
        b1 += queue.get()
        print(np.sum(np.abs(b1)))
    for job in jobs:
        try:
            job.join()
        except Exception as e:
            print(e)

    ft = time.time() - st
    print("finish")
    print("Elapsed time: {0:.2f} s".format(ft))

    rho = np.zeros((N, N), dtype=float)
    for ii in range(len(sprange)):
        rho += x*x + y*y <= R*R - sprange[ii]*sprange[ii]

    plt.figure(figsize=(12, 5), dpi=100)
    plt.subplot(121)
    plt.imshow(rho, extent=[sprange.min(), sprange.max(), sprange.min(), sprange.max()])
    plt.xlim(-R*1.5, R*1.5)
    plt.ylim(-R*1.5, R*1.5)
    plt.subplot(122)
    plt.imshow(np.abs(b1), extent=[q.min(), q.max(), q.min(), q.max()])
    plt.xlim(-3*x0/R, 3*x0/R)
    plt.ylim(-3*x0/R, 3*x0/R)
    plt.show()