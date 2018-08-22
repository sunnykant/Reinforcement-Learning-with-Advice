import numpy as np
import matplotlib.pyplot as plt

fname = ('trdata-7-10-16-42-12.csv')
pathname = ('/home/starshipcrew/results/noadvice/')
epi, score = np.loadtxt(pathname+fname,unpack = 1, delimiter = ',' )


movavg = 100
avg = []
avgepisode = []
i = movavg-1
while i <= len(epi):
	avg.append(sum(score[i+1-movavg:i]) / movavg)
	avgepisode.append(i)
	i = i+movavg

plt.plot(avgepisode, avg , 'b-')

plt.xlabel('Training Episodes')
plt.ylabel('Reward')
plt.title('Rewards accumulated - No Advice')
plt.axis([0, len(epi), -1000,1000 ])
plt.grid(True)
plt.show()
