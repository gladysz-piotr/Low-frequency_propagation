import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import patches as pat
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib import rc
from matplotlib import animation
from scipy.integrate import odeint
from scipy.special import jv as J
from tqdm import tqdm
from scipy.fftpack import fft

def openPickle(filename):
	with open(filename, 'rb') as f:
		var=pickle.load(f)
		f.close
	return var

def init():
	u_plot.set_ydata(np.ma.array(X, mask=True))
	return u_plot, 

def animate(i, *fargs):
	u_plot.set_ydata(u_now[i])
	return u_plot,
	
def fourierTransform(signal, N, step):
	ft_signal=fft(signal)
	#creating frequency axis
	w=np.linspace(0, np.pi/step, N//2)
	return ft_signal, w	
	
if __name__=="__main__":
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	rc('font', size=21)

	c=137.
	slope=1.e-10
	alpha1=1**2*6.5e-23 #1, 2, 3, 4
	alpha2=3**2*6.5e-23
	E=3.e-7


##### temporal and spatial comparison (publication fig1) #####
	'''
	x0=-1e11
	dt=5e5
	dx=c*dt
	n=c*dt/dx
	X=np.arange(0, 4.e10, dx)
	T=np.arange(0, 5e9, dt)
	T_c=T[:5000] #5000
	L=len(X)//2
	
	Y=openPickle('53cm_sample/delta/E3e-7_delta0_N12.pkl')
	time=np.array([Y[i][L] for i in xrange(0, len(T_c))])
	space=np.array([Y[2500][j] for j in xrange(0, len(X))]) #2085 max
	
	fig, axs=plt.subplots(3, figsize=[10, 15])	
	
	axs[0].plot(T_c, time, linewidth=2, color='blue')
	axs[0].text(0.1e9, 2.45e-11, 'a.', ha='center', va='center',\
		fontsize=20, color='black')	
	axs[0].text(0.15e9, 0.2e-11, 'signal', ha='center', va='center',\
		fontsize=20, color='blue')	
	axs[0].text(0.15e9, -2.5e-11, 'drive', ha='center', va='center',\
		fontsize=20, color='black')	
	x_formatter = FixedFormatter(['0', '25', '50', '75'])
	x_locator = FixedLocator([0, 1.033e9, 2.066e9, 3.099e9])	
	y_formatter = FixedFormatter(['-1.5', '-1.0', '-0.5', '0', '0.5',\
		'1.0', '1.5'])
	y_locator = FixedLocator([-2.92e-11, -1.946e-11, -0.973e-11, 0.0,\
		0.973e-11, 1.946e-11, 2.92e-11])
	axs[0].xaxis.set_major_formatter(x_formatter)
	axs[0].xaxis.set_major_locator(x_locator)
	axs[0].yaxis.set_major_formatter(y_formatter)
	axs[0].yaxis.set_major_locator(y_locator)
	axs[0].set_xlabel('Time $t$ [ns]')
	axs[0].set_ylabel(r'$E_\mathrm{signal}(t)$ [$\times10^{-1}$ \
		$\frac{\mathrm{V}}{\mathrm{cm}}$]')
	axs2=axs[0].twinx()
	y_formatter2 = FixedFormatter(['-15', '-10', '-5', '0', '5', '10', '15'])
	y_locator2 = FixedLocator([-2.92e-7, -1.946e-7, -0.973e-7, 0.0,\
		0.973e-7, 1.946e-7, 2.92e-7])	
	axs2.plot(T_c, [(E/np.pi)*(np.arctan(-(X[L]-c*t-x0)*slope)+(np.pi/2))\
		for t in T_c], linewidth='2.0', color='black')	
	axs2.tick_params(axis='y', labelcolor='black')	
	axs2.yaxis.set_major_formatter(y_formatter2)
	axs2.yaxis.set_major_locator(y_locator2)
	axs2.set_ylabel(r'$\mathcal{E}(t)$ [$\times 10^{2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]',\
		color='black')	
	fig.tight_layout()		
			
	axs[1].plot(X, space, linewidth=2, color='blue')
	axs[1].text(0.18e10, 2.45e-11, 'b.', ha='center', va='center',\
		fontsize=20, color='black')	
	x_formatter = FixedFormatter(['-0.53', '0', '0.53', '1.06', '1.5'])
	x_locator = FixedLocator([0, 1e10, 2e10, 2.8302e10, 3.7736e10])	
	axs[1].xaxis.set_major_formatter(x_formatter)
	axs[1].xaxis.set_major_locator(x_locator)
	axs[1].yaxis.set_major_formatter(y_formatter)
	axs[1].yaxis.set_major_locator(y_locator)	
	axs[1].set_xlabel('Position $z$ [m]')
	axs[1].set_ylabel(r'$E_\mathrm{signal}(z)$ [$\times10^{-1}$ \
		$\frac{\mathrm{V}}{\mathrm{cm}}$]')
	rec=pat.Rectangle((1e10,-3e-11), 1e10, 6e-11)
	rec.set_alpha(0.5)
	rec.set_color('grey')
	axs[1].add_patch(rec)

	Y2=openPickle('53cm_sample/delta/E3e-7_delta7e-8_N12.pkl')
	time2=np.array([Y2[i][L] for i in xrange(0, len(T_c))])
	space2=np.array([Y2[2500][j] for j in xrange(0, len(X))])
	
	y_formatter = FixedFormatter(['-1.0', '-0.5', '0', '0.5', '1.0'])
	y_locator = FixedLocator([-1.946e-12, -0.973e-12, 0.0, 0.973e-12,\
		1.946e-12])	
	axs[2].plot(X, space2, linewidth=2, color='blue')
	axs[2].text(0.18e10, 1.15e-12, 'c.', ha='center', va='center',\
		fontsize=20, color='black')	
	x_formatter = FixedFormatter(['-0.53', '0', '0.53', '1.06', '1.5'])
	x_locator = FixedLocator([0, 1e10, 2e10, 2.8302e10, 3.7736e10])	
	axs[2].xaxis.set_major_formatter(x_formatter)
	axs[2].xaxis.set_major_locator(x_locator)
	axs[2].yaxis.set_major_formatter(y_formatter)
	axs[2].yaxis.set_major_locator(y_locator)	
	axs[2].set_xlabel('Position $z$ [m]')
	axs[2].set_ylabel(r'$E_\mathrm{signal}(z)$ [$\times10^{-2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]')
	rec=pat.Rectangle((1e10,-3e-11), 1e10, 6e-11)
	rec.set_alpha(0.5)
	rec.set_color('grey')
	axs[2].add_patch(rec)
	
	plt.subplots_adjust(left= 0.11, right= 0.9, bottom=0.08, hspace=0.3)	
	plt.savefig('53cm_sample/delta/signal.png', transparent=True)
	plt.show()
	'''
##### end #####


##### gamma gamma_dec comparison (publication fig2)#####
	'''
	dt=1e6
	dx=c*dt
	n=c*dt/dx
	X=np.arange(0, 4.e10, dx)
	T=np.arange(0, 5e9, dt)
	x0=-1e11
	T_c=T[:] #5000
	L=len(X)//2
	
	x_formatter=FixedFormatter(['0', '25', '50', '75', '100'])
	x_locator=FixedLocator([0, 1.033e9, 2.066e9, 3.099e9, 4.132e9])	
	y_formatter_drive = FixedFormatter(['-15', '-10', '-5', '0', '5',\
		'10', '15'])
	y_locator_drive = FixedLocator([-2.92e-7, -1.946e-7, -0.973e-7,\
		0.0, 0.973e-7, 1.946e-7, 2.92e-7])	
	
	Y=openPickle('53cm_sample/gamma/gamma_dec.pkl')
	time=np.array([Y[i][L] for i in xrange(0, len(T_c))])
	
	fig, axs=plt.subplots(3, figsize=[10, 15])	
	
	y_formatter = FixedFormatter(['-7.5', '-5.0', '-2.5', '0', '2.5',\
		'5.0', '7.5'])
	y_locator = FixedLocator([-2.92e-11/2, -1.946e-11/2, -0.973e-11/2,\
		0.0, 0.973e-11/2, 1.946e-11/2, 2.92e-11/2])
	
	axs[0].plot(T_c, time, linewidth=2, color='blue',\
		label=r'$\gamma_{se}=1e-9$')
	axs[0].text(0.2e9, 2.45e-11/2, 'a.', ha='center', va='center',\
		fontsize=20, color='black')
	axs[0].text(0.4e9, 0.2e-11/2, 'signal', ha='center', va='center',\
		fontsize=20, color='blue')	
	axs[0].text(0.4e9, -2.25e-11/2, 'drive', ha='center', va='center',\
		fontsize=20, color='black')
	axs[0].xaxis.set_major_formatter(x_formatter)
	axs[0].xaxis.set_major_locator(x_locator)
	axs[0].yaxis.set_major_formatter(y_formatter)
	axs[0].yaxis.set_major_locator(y_locator)
	axs[0].set_xlabel('Time $t$ [ns]')
	axs[0].set_ylabel(r'$E_\mathrm{signal}(t)$ [$\times10^{-2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]')
	axs2=axs[0].twinx()
	axs2.plot(T_c, [(E/np.pi)*(np.arctan(-(X[L]-c*t-x0)*slope)+(np.pi/2))\
		for t in T_c], linewidth='2.0', color='black')	
	axs2.tick_params(axis='y', labelcolor='black')	
	axs2.yaxis.set_major_formatter(y_formatter_drive)
	axs2.yaxis.set_major_locator(y_locator_drive)
	axs2.set_ylabel(r'$\mathcal{E}(t)$ [$\times 10^{2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]',\
		color='black')	
	fig.tight_layout()		
	
	Y=openPickle('53cm_sample/gamma/gamma.pkl')
	time=np.array([Y[i][L] for i in xrange(0, len(T_c))])
	
	y_formatter = FixedFormatter(['-3.0', '-2.0', '-1.0', '0', '1.0',\
		'2.0', '3.0'])
	y_locator = FixedLocator([-2.92e-12*2, -1.946e-12*2, -0.973e-12*2,\
		0.0, 0.973e-12*2, 1.946e-12*2, 2.92e-12*2])	
	
	axs[1].plot(T_c, time, linewidth=2, color='blue',\
		label=r'$\gamma_{dec}=1e-9$')		
	axs[1].text(0.2e9, 2.45e-11/5, 'b.', ha='center', va='center',\
		fontsize=20, color='black')
	axs[1].xaxis.set_major_formatter(x_formatter)
	axs[1].xaxis.set_major_locator(x_locator)
	axs[1].yaxis.set_major_formatter(y_formatter)
	axs[1].yaxis.set_major_locator(y_locator)
	axs[1].set_xlabel('Time $t$ [ns]')
	axs[1].set_ylabel(r'$E_\mathrm{signal}(t)$ [$\times10^{-2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]')
	axs2=axs[1].twinx()
	axs2.plot(T_c, [(E/np.pi)*(np.arctan(-(X[L]-c*t-x0)*slope)+(np.pi/2))\
		for t in T_c], linewidth='2.0', color='black')	
	axs2.tick_params(axis='y', labelcolor='black')	
	axs2.yaxis.set_major_formatter(y_formatter_drive)
	axs2.yaxis.set_major_locator(y_locator_drive)
	axs2.set_ylabel(r'$\mathcal{E}(t)$ [$\times 10^{2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]',\
		color='black')	
	fig.tight_layout()	
	
	Y=openPickle('53cm_sample/gamma/gamma_gamma_dec.pkl')
	time=np.array([Y[i][L] for i in xrange(0, len(T_c))])
	
	y_formatter = FixedFormatter(['-2.0', '-1.0', '0', '1.0', '2.0'])
	y_locator = FixedLocator([-1.946e-12*2, -0.973e-12*2, 0.0,\
		0.973e-12*2, 1.946e-12*2])	
	
	axs[2].plot(T_c, time, linewidth=2, color='blue',\
		label=r'$\gamma_{dec}+\gamma_{se}$')	
	axs[2].text(0.2e9, 2.45e-11/7.5, 'c.', ha='center', va='center',\
		fontsize=20, color='black')
	axs[2].xaxis.set_major_formatter(x_formatter)
	axs[2].xaxis.set_major_locator(x_locator)
	axs[2].yaxis.set_major_formatter(y_formatter)
	axs[2].yaxis.set_major_locator(y_locator)	
	axs[2].set_xlabel('Time $t$ [ns]')
	axs[2].set_ylabel(r'$E_\mathrm{signal}(t)$ [$\times10^{-2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]')
	axs2=axs[2].twinx()
	axs2.plot(T_c, [(E/np.pi)*(np.arctan(-(X[L]-c*t-x0)*slope)+(np.pi/2))\
		for t in T_c], linewidth='2.0', color='black')	
	axs2.tick_params(axis='y', labelcolor='black')	
	axs2.yaxis.set_major_formatter(y_formatter_drive)
	axs2.yaxis.set_major_locator(y_locator_drive)	
	axs2.set_ylabel(r'$\mathcal{E}(t)$ [$\times 10^{2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]',\
		color='black')	
	fig.tight_layout()
	
	plt.subplots_adjust(left= 0.11, right= 0.9, bottom=0.08, hspace=0.3)
	plt.savefig('53cm_sample/gamma/gammas.png', transparent=True)
	plt.show()	
	'''
##### end #####


##### back action (publication fig3) #####	
	'''
	x0=-1e11
	dt=5e6
	dx=c*dt
	n=c*dt/dx
	X=np.arange(0, 4.e10, dx)
	T=np.arange(0, 5e10, dt)
	T=T[:5000]
	L=len(X)//2
	
	Y1=openPickle('53cm_sample/backaction/backaction_E1e-7_no_gamma_N12.pkl')
	Y1g=openPickle('53cm_sample/backaction/backaction_E1e-7_gamma_N12.pkl')
	Y2=openPickle('53cm_sample/backaction/backaction_E1e-7_no_gamma_N11.pkl')
	Y2g=openPickle('53cm_sample/backaction/backaction_E1e-7_gamma_N11.pkl')
	Y3=openPickle('53cm_sample/backaction/backaction_E1e-7_no_gamma_N10.pkl')
	Y3g=openPickle('53cm_sample/backaction/backaction_E1e-7_gamma_N10.pkl')
	time1=np.array([Y1[i][L] for i in xrange(0, len(T))])
	time1g=np.array([Y1g[i][L] for i in xrange(0, len(T))])
	time2=np.array([Y2[i][L] for i in xrange(0, len(T))])
	time2g=np.array([Y2g[i][L] for i in xrange(0, len(T))])
	time3=np.array([Y3[i][L] for i in xrange(0, len(T))])
	time3g=np.array([Y3g[i][L] for i in xrange(0, len(T))])
	
	fig, axs=plt.subplots(3, figsize=[10, 15])	
	
	axs[0].plot(T, time1, linewidth=2, color='black', alpha=0.5)
	axs[0].text(0.1e10, 2.45e-11, 'a.', ha='center', va='center',\
		fontsize=20, color='black')
	axs[0].plot(T, time1g, linewidth=2, color='blue')
	axs[0].text(2.15e10, 2.1e-11, r'$N=6.7\times 10^{12}$', ha='center',\
		va='center', fontsize=21, bbox=dict(boxstyle="square",\
		fc=(1., 1., 1.), ec=(0., 0., 0.)))
	axs[1].plot(T, time2, linewidth=2, color='black', alpha=0.5)
	axs[1].text(0.1e10, 2.45e-10, 'b.', ha='center', va='center',\
		fontsize=20, color='black')
	axs[1].plot(T, time2g, linewidth=2, color='blue')
	axs[1].text(2.15e10, 2.1e-10, r'$N=6.7\times 10^{13}$', ha='center',\
		va='center', fontsize=21, bbox=dict(boxstyle="square",\
		fc=(1., 1., 1.), ec=(0., 0., 0.)))	
	axs[2].plot(T, time3, linewidth=2, color='black', alpha=0.5)
	axs[2].text(0.1e10, 2.45e-9, 'c.', ha='center', va='center',\
		fontsize=20, color='black')
	axs[2].plot(T, time3g, linewidth=2, color='blue')
	axs[2].text(2.15e10, 2.1e-9, r'$N=6.7\times 10^{14}$', ha='center',\
		va='center', fontsize=21, bbox=dict(boxstyle="square",\
		fc=(1., 1., 1.), ec=(0., 0., 0.)))	

	x_formatter = FixedFormatter(['0', '250', '500', '750', '1000'])
	x_locator = FixedLocator([0, 1.033e10, 2.066e10, 3.099e10, 4.132e10])	
	for i in xrange(0,3):
		y_formatter = FixedFormatter(['-1.5', '-1.0', '-0.5', '0', '0.5',\
			'1.0', '1.5'])
		y_locator = FixedLocator([-2.92e-11*10**i, -1.946e-11*10**i,\
			-0.973e-11*10**i, 0.0, 0.973e-11*10**i, 1.946e-11*10**i,\
			2.92e-11*10**i])		
		axs[i].xaxis.set_major_formatter(x_formatter)
		axs[i].xaxis.set_major_locator(x_locator)
		axs[i].yaxis.set_major_formatter(y_formatter)
		axs[i].yaxis.set_major_locator(y_locator)		
		axs[i].set_xlabel('Time $t$ [ns]')
		if i==0:
			axs[i].set_ylabel(r'$E_\mathrm{signal}(t)$ [$\times10^{-1}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]')	
		if i==1:
			axs[i].set_ylabel(r'$E_\mathrm{signal}(t)$ [$\frac{\mathrm{V}}{\mathrm{cm}}$]')	
		if i==2:	
			axs[i].set_ylabel(r'$E_\mathrm{signal}(t)$ [$\times10$ $\frac{\mathrm{V}}{\mathrm{cm}}$]')	
	plt.subplots_adjust(left= 0.11, right= 0.9, bottom=0.08, hspace=0.3)	
	plt.savefig('53cm_sample/backaction/backaction.png', transparent=True)
	plt.show()
	'''
##### end #####


##### fourier transformation for several detunings (publication fig 4)#####
	'''
	dt=1e6
	T=np.arange(0, 1e10, dt)	
	c=['blue', 'green', 'red', 'cyan', 'purple']
	fig,ax=plt.subplots(figsize=[10, 5])
	O=0.5*2.98e-7
	i=-1
	for delta in [0.0, 2e-8, 5e-8, 7e-8, 1e-7]:
		i+=1
		end=openPickle('53cm_sample/delta_'+str(delta)+'.pkl')[600:]
		fourier=fourierTransform(end, len(T[600:]), dt)
		M=max(abs(fourier[0][1:len(fourier[0])//2]))
		ax.plot(fourier[1][1:]*6.58e15, np.abs(fourier[0][1:len(fourier[0])\
		//2])/M, linewidth=2.0, label=r'$\delta=$'+\
			str(np.round(delta*6.58e15*1e-9, 2)), color=c[i])
		ax.set_xlim([1.85e9, 2.25e9])
		result = np.where(np.abs(fourier[0][1:len(fourier[0])//2])/M ==\
			np.amax(np.abs(fourier[0][1:len(fourier[0])//2])/M))
	x_formatter = FixedFormatter(['1.8', '1.9', '2.0', '2.1', '2.2'])
	x_locator = FixedLocator([1.8e9, 1.9e9, 2e9, 2.1e9, 2.2e9])	
	ax.xaxis.set_major_formatter(x_formatter)
	ax.xaxis.set_major_locator(x_locator)
	ax.set_xlabel(r'Angular frequency $\Omega$ [GHz]')
	ax.set_ylabel('Normalized FFT')
	
	size=5e5
	Recs=[pat.Rectangle((1.960e9,0), size, 1),\
	pat.Rectangle((1.966e9,0), size, 1),\
	pat.Rectangle((1.988e9,0), size, 1),\
	pat.Rectangle((2.014e9,0), size, 1),\
	pat.Rectangle((2.068e9,0), size, 1)]
	i=-1
	for rec in Recs:
		i+=1
		rec.set_color(c[i])
		rec.set_alpha(0.6)
		ax.add_patch(rec)
	plt.legend(title='Detuning in GHz')	
	plt.subplots_adjust(left= 0.11, right= 0.95, bottom=0.15)
	plt.savefig('53cm_sample/fft.png', transparent=True)
	plt.show()
	'''
##### end #####


##### gaussian impulse temporal and spatial (publication fig 5) #####
	'''
	dt=1e6
	dx=c*dt
	n=c*dt/dx
	T=np.arange(0, 5e9, dt)
	X=np.arange(0, 4.e10, dx)
	x0=-3.4e11
	T_c=T[:] #5000
	L=len(X)//2
	
	Y1=openPickle('53cm_sample/gauss/E3e-7_N1e-12_1.5au.pkl')
	Y2=openPickle('53cm_sample/gauss/E3e-7_N1e-12_0.5au.pkl')
	
	time1=np.array([Y1[i][L] for i in xrange(0, len(T_c))])
	time2=np.array([Y2[i][L] for i in xrange(0, len(T_c))])

	fig, axs=plt.subplots(3, figsize=[10, 15])	
	x_formatter=FixedFormatter(['0', '25', '50', '75', '100'])
	x_locator=FixedLocator([0, 1.033e9, 2.066e9, 3.099e9, 4.132e9])	
	y_formatter = FixedFormatter(['-1.0', '-0.5', '0', '0.5', '1.0'])
	y_locator = FixedLocator([-1.946e-11, -0.973e-11, 0.0, 0.973e-11,\
		1.946e-11])
	y_formatter_drive = FixedFormatter(['-15', '-10', '-5', '0', '5',\
		'10', '15'])
	y_locator_drive = FixedLocator([-2.92e-7, -1.946e-7, -0.973e-7,\
		0.0, 0.973e-7, 1.946e-7, 2.92e-7])
	
	axs[0].plot(T_c, time1, linewidth=2, color='blue')	
	axs[0].text(0.2e9, 1.65e-11, 'a.', ha='center', va='center',\
		fontsize=20, color='black')	
	axs[0].text(0.3e9, 0.2e-11, 'signal', ha='center', va='center',\
		fontsize=20, color='blue')	
	axs[0].text(0.3e9, -1.8e-11, 'drive', ha='center', va='center',\
		fontsize=20, color='black')		
	axs[0].xaxis.set_major_formatter(x_formatter)
	axs[0].xaxis.set_major_locator(x_locator)
	axs[0].yaxis.set_major_formatter(y_formatter)
	axs[0].yaxis.set_major_locator(y_locator)	
	axs[0].set_xlabel('Time $t$ [ns]')
	axs[0].set_ylabel(r'$E_\mathrm{signal}(t)$ [$\times10^{-1}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]')
	axs2=axs[0].twinx()
	axs2.plot(T_c, [E*np.exp(-alpha1*(X[L]-x0-c*t)**2)\
		for t in T_c], linewidth='2.0', color='black')
	axs2.yaxis.set_major_formatter(y_formatter_drive)
	axs2.yaxis.set_major_locator(y_locator_drive)		
	axs2.tick_params(axis='y', labelcolor='black')	
	axs2.set_ylabel(r'$\mathcal{E}(t)$ [$\times 10^{2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]',\
		color='black')	
	fig.tight_layout()
	
	y_formatter = FixedFormatter(['-2.0','-1.0','0','1.0', '2.0'])
	y_locator = FixedLocator([-3.89e-11, -1.946e-11, 0.0, 1.946e-11, 3.89e-11])		
	axs[1].plot(T_c, time2, linewidth=2, color='green')	
	axs[1].text(0.2e9, 3.8e-11, 'b.', ha='center', va='center',\
		fontsize=20, color='black')	
	axs[1].text(0.3e9, 0.5e-11, 'signal', ha='center', va='center',\
		fontsize=20, color='green')	
	axs[1].text(0.3e9, -4.4e-11, 'drive', ha='center', va='center',\
		fontsize=20, color='black')	
	axs[1].xaxis.set_major_formatter(x_formatter)
	axs[1].xaxis.set_major_locator(x_locator)
	axs[1].yaxis.set_major_formatter(y_formatter)
	axs[1].yaxis.set_major_locator(y_locator)	
	axs[1].set_xlabel('Time $t$ [ns]')
	axs[1].set_ylabel(r'$E_\mathrm{signal}(t)$ [$\times10^{-1}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]')
	axs2=axs[1].twinx()
	axs2.plot(T_c, [E*np.exp(-alpha2*(X[L]-x0-c*t)**2)\
		for t in T_c], linewidth='2.0', color='black')	
	axs2.yaxis.set_major_formatter(y_formatter_drive)
	axs2.yaxis.set_major_locator(y_locator_drive)	
	axs2.tick_params(axis='y', labelcolor='black')
	axs2.set_ylabel(r'$\mathcal{E}(t)$ [$\times 10^{2}$ $\frac{\mathrm{V}}{\mathrm{cm}}$]',\
		color='black')
	fig.tight_layout()

	x_formatter = FixedFormatter(['0', '0.5', '1.0', '1.5', '2.0', '2.5'])
	x_locator = FixedLocator([0, 0.5e9, 1e9, 1.5e9, 2e9, 2.5e9])	
	
	fourier1=fourierTransform(time1, len(T), dt)
	M1=max(abs(fourier1[0][1:len(fourier1[0])//2]))
	result1 = np.where(np.abs(fourier1[0][1:len(fourier1[0])//2])/M1 ==\
	np.amax(np.abs(fourier1[0][1:len(fourier1[0])//2])/M1))
	fourier2=fourierTransform(time2, len(T), dt)
	M2=max(abs(fourier2[0][1:len(fourier2[0])//2]))
	result2 = np.where(np.abs(fourier2[0][1:len(fourier2[0])//2])/M2 ==\
	np.amax(np.abs(fourier2[0][1:len(fourier2[0])//2])/M2))
	
	axs[2].plot(fourier1[1][1:]*6.58e15, np.abs(fourier1[0][1:len(fourier1[0])\
	//2])/M1, linewidth=2.0, label='wider')
	axs[2].plot(fourier2[1][1:]*6.58e15, np.abs(fourier2[0][1:len(fourier2[0])\
	//2])/M2, linewidth=2.0, label='narrower')
	axs[2].text(1.e8, 0.85, 'c.', ha='center', va='center', fontsize=20,\
		color='black')	
	axs[2].xaxis.set_major_formatter(x_formatter)
	axs[2].xaxis.set_major_locator(x_locator)
	axs[2].set_xlabel(r'Angular frequency $\Omega$ [GHz]')
	axs[2].set_ylabel('Normalized FFT')
	axs[2].set_xlim([0,3.15e9])
	axs[2].legend()
	
	plt.subplots_adjust(left= 0.11, right= 0.9, bottom=0.08, hspace=0.3)	
	plt.savefig('53cm_sample/gauss/gaussy.png', transparent=True)
	plt.show()
	'''
##### end #####
	
	
##### gaussian gamma no gamma comparison #####
	'''
	dt=1e6
	#dt=1e6# for better
	dx=c*dt
	n=c*dt/dx
	T=np.arange(0, 5e9, dt)
	X=np.arange(0, 4.e10, dx)
	x0=-3.4e11
	T_c=T[:] #5000
	L=len(X)//2
	
	Y1=openPickle('53cm_sample/gauss/E3e-7_N1e-12_0.5au.pkl')
	Y2=openPickle('53cm_sample/gauss/E3e-7_N1e-12_0.5au_no_gamma.pkl')
	
	time1=np.array([Y1[i][L] for i in xrange(0, len(T_c))])
	time2=np.array([Y2[i][L] for i in xrange(0, len(T_c))])

	fig, ax=plt.subplots(1, figsize=[10,10])
	ax.plot(T, time2, label='no gamma', color='green')
	ax.plot(T, time1, label='gamma', color='blue')
	ax.set_ylim([0, 8e-11])
	ax2=ax.twinx()
	ax2.plot(T, [E*np.exp(-alpha1*(X[L]-x0-c*t)**2)\
		for t in T_c], color='black')
	ax2.tick_params(axis='y', labelcolor='black')
	fig.tight_layout()
	ax.legend()
	plt.savefig('53cm_sample/gauss/no_gamma_narrower_2.png', transparent=False)
	plt.show()
	'''
##### end #####
	

##### length to concentration comparison #####
	'''
	dt=1e6
	dx=c*dt
	n=c*dt/dx
	T=np.arange(0, 5e9, dt)
	Y1=openPickle('different_lengths/E1e-7_N4e-12_X26cm.pkl')
	Y2=openPickle('different_lengths/E1e-7_N2e-12_X53cm.pkl')
	Y3=openPickle('different_lengths/E1e-7_N1e-12_X106cm.pkl')
	L1=len(Y1[0])//2
	L2=len(Y2[0])//2
	L3=len(Y3[0])//2
	print L1, L2, L3
	time1=np.array([Y1[i][L1] for i in xrange(0, len(T))])
	time2=np.array([Y2[i][L2] for i in xrange(0, len(T))])
	time3=np.array([Y3[i][L3] for i in xrange(0, len(T))])
	
	plt.plot(T, time1, label='26 cm')
	plt.plot(T, time2, label='53 cm')
	plt.plot(T, time3, label='106 cm')
	plt.legend()
	plt.savefig('different_lengths/compare.png', transparent=False)
	plt.show()
	'''
##### end #####		
		
		
##### animation #####	
	'''
	dt=5e5
	dx=c*dt
	n=c*dt/dx
	X=np.arange(0, 4.e10, dx)
	T=np.arange(0, 5e9, dt)	
	fig, ax = plt.subplots()
	
	u_now=openPickle('53cm_sample/temp.pkl')	
	ax.set_ylim([-5e-10, 5e-10])
	u_plot, = ax.plot(X, u_now[0], c='black', linewidth='3.0')
	anim = animation.FuncAnimation(fig, animate, frames=None, init_func=init,\
		fargs=(u_now), interval=40, blit=True, repeat=False)	
	plt.show()	
	'''
##### end #####



