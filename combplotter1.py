import numpy as np
import matplotlib.pyplot as plt

fname1 = ('trdata_win8-22-15-22-45.csv')
pathname1 = ('/home/sunny/Desktop/intern_30_07/results/noadvice/')


fname2 = ('trdata_win8-22-15-36-51.csv')
pathname2 = ('/home/sunny/Desktop/intern_30_07/results/singleadvice/')

fname3 = ('trdata_win8-22-15-39-26.csv')
pathname3 = ('/home/sunny/Desktop/intern_30_07/results/multiadvice/')


#fname3 = ('trdata-7-11-17-47-18.csv')
#pathname3 = ('/home/starshipcrew/results/multiadvice/')

epi1, score1 = np.loadtxt(pathname1+fname1,unpack = 1, delimiter = ',' )
epi2, score2 = np.loadtxt(pathname2+fname2,unpack = 1, delimiter = ',' )
epi3, score3 = np.loadtxt(pathname3+fname3,unpack = 1, delimiter = ',' )


movavg = 25
avg1 = []
avg2 = []
avg3 = []
avgepisode = []
i = movavg-1
while i <= len(epi1):
	avg1.append(sum(score1[i+1-movavg:i]) / movavg)
	avg2.append(sum(score2[i+1-movavg:i]) / movavg)
	avg3.append(sum(score3[i+1-movavg:i]) / movavg)
	avgepisode.append(i)
	i = i+movavg

print (len(avg1),len(avg2),len(avg3))


plt.figure(1)

#plt.plot(avgepisode, avg1 , 'b-',avgepisode, avg2 , 'k-',avgepisode, avg3, 'r-', linewidth = 0.8)
plt.plot(avgepisode, avg1, 'k--', linewidth = 0.8,label="no-advice")
plt.plot(avgepisode, avg2, 'b-', linewidth = 0.8,label="with-advice")
plt.plot(avgepisode, avg3, 'r-', linewidth = 0.8, label = 'Multiple advice')
plt.xlabel('Training Episodes',size = 'larger')
plt.ylabel('Wining rate', size = 'larger')
plt.title('Wining rate over last 10 Episodes', size = 'larger')
plt.axis([0, len(epi1), -1,1 ])
plt.grid(True)
plt.legend()



# Create two subplots sharing y axis
'''
fig, (ax1, ax2) = plt.subplots(2, sharey=True)

ax1.plot(x1, y1, 'ko-')
ax1.set(title='A tale of 2 subplots', ylabel='Damped oscillation')

ax2.plot(x2, y2, 'r.-')
ax2.set(xlabel='time (s)', ylabel='Undamped')

plt.grid(True)

'''
plt.show()

