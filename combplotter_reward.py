import numpy as np
import matplotlib.pyplot as plt
fname1 = ('trdata-8-22-2-28-20.csv')
pathname1 = ('/home/sunny/Desktop/intern_30_07/results/noadvice/')


fname2 = ('trdata-8-22-1-9-44.csv')
pathname2 = ('/home/sunny/Desktop/intern_30_07/results/singleadvice/')

fname3 = ('trdata-8-22-3-13-13.csv')
pathname3 = ('/home/sunny/Desktop/intern_30_07/results/multiadvice/')


#fname3 = ('trdata-7-11-17-47-18.csv')
#pathname3 = ('/home/starshipcrew/results/multiadvice/')

epi1, score1 = np.loadtxt(pathname1+fname1,unpack = 1, delimiter = ',' )
epi2, score2 = np.loadtxt(pathname2+fname2,unpack = 1, delimiter = ',' )
epi3, score3 = np.loadtxt(pathname3+fname3,unpack = 1, delimiter = ',' )
#epi3, score3 = np.loadtxt(pathname3+fname3,unpack = 1, delimiter = ',' )


avg1 = []
avg2 = []
avg3 = []
#avg3 = []
avgepisode = []
i = 0
while i < len(epi1):
	avg1.append(sum(score1[0:i]) / (i+1))
	avg2.append(sum(score2[0:i]) / (i+1))
	avg3.append(sum(score3[0:i]) / (i+1))
	avgepisode.append(i+1)
	i = i+1

print (len(avg1),len(avg2),len(avg3))


plt.figure(1)

plt.subplot(211)
#plt.plot(avgepisode, avg1 , 'b-',avgepisode, avg2 , 'k-',avgepisode, avg3, 'r-', linewidth = 0.8)
plt.plot(avgepisode, avg1, 'k--', linewidth = 0.8,label="No-advice")
plt.plot(avgepisode, avg2, 'b-', linewidth = 0.8,label="Single-advice")
#plt.plot(avgepisode, avg3, 'r-', linewidth = 0.8, label = 'Multiple advice')
plt.xlabel('Training Episodes',size = 'larger')
plt.ylabel('Reward', size = 'larger')
plt.title('Average Rewards over Episodes', size = 'larger')
plt.axis([0, len(epi1), -500,1500 ])
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

