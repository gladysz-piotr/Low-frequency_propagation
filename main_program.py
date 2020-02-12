# -*- coding:utf8 -*-
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
from scipy.integrate import odeint
from scipy.special import jv as J
from tqdm import tqdm
from multiprocessing import Process, Queue

def System(state, time, x0, x, t, c, delta, gamma, gamma_dec, E, omega,\
		d_eg, d_ee, d_gg, hbar, slope, Om):
	'''
	Function used to define Bloch equations. Standard approach with method 
	odeint from scipy.intgrate.
	'''
	alpha=slope
	
	#if gauss=False for smooth step function as an envelope, True for Gaussian
	gauss=True
	if gauss:
		#kappa
		kappa=((d_ee-d_gg)/(hbar*omega))*(E*np.exp(-alpha*(x-x0-c*t)**2))
		#d/dt kappa
		kappa_d=((d_ee-d_gg)/(hbar*omega))*\
		2*E*alpha*c*(x-x0-c*t)*np.exp(-alpha*(x-x0-c*t)**2)
	else:	
		#kappa
		kappa=((d_ee-d_gg)/(hbar*omega))*(E/np.pi)*\
			(np.arctan(-(x-c*t-x0)*slope)+(np.pi/2))
		#d/dt kappa
		kappa_d=((d_ee-d_gg)/(hbar*omega))*\
			(E*slope*c)/(np.pi*(1+(slope*(x-c*t-x0))**2))
	

	rho_ee, sigma_Im, sigma_Re = state
	d_rho_ee=2.*((omega/(d_ee-d_gg))+Om/hbar)*J(1, kappa)*(np.real(d_eg)\
		*sigma_Im-np.imag(d_eg)*sigma_Re)-2*gamma*rho_ee
	d_sigma_Re=delta*sigma_Im-Om*((d_ee-d_gg)/hbar)*sigma_Im\
		-((omega/(d_ee-d_gg))+Om/hbar)*J(1, kappa)*np.imag(d_eg)*\
		(1.- 2.*rho_ee)-(gamma+gamma_dec)*sigma_Re
	d_sigma_Im=(-delta+kappa_d)*sigma_Re+Om*((d_ee-d_gg)/hbar)*sigma_Re\
		+((omega/(d_ee-d_gg))+Om/hbar)*J(1, kappa)*np.real(d_eg)*\
		(1.- 2.*rho_ee)-(gamma+gamma_dec)*sigma_Im
	
	return [d_rho_ee, d_sigma_Im, d_sigma_Re]	
	
def source(x0, X, t, c, slope, rho_prev_prev, rho_prev, rho_now, dt, N,\
		d_ee, d_gg, d_ge, delta, gamma, gamma_dec, hbar, omega, E):
	'''
	Function used to calculate source term at each point of the sample.
	'''
	alpha=slope
	def Diff(i,j):
		'''
		First order differential with accuracy dt^2.
		'''
		return (rho_prev_prev[j][i]-rho_now[j][i])/(2*dt)
	
	def sDiff(i,j):
		'''
		Second order differential with accuracy dt^2.
		'''
		return (rho_prev_prev[j][i]+rho_now[j][i]-2*rho_prev[j][i])/(dt**2)
		
	sour=[]
	
	#if gauss=False for smooth step function as an envelope, True for Gaussian
	gauss=True
	d_eg=np.conj(d_ge)
	
	for j in xrange(0, len(rho_now)):
		if gauss:
			#kappa
			kappa=((d_ee-d_gg)/(hbar*omega))*\
				(E*np.exp(-alpha*(X[j]-x0-c*t)**2))
			#d/dt kappa
			kappa_d=((d_ee-d_gg)/(hbar*omega))*\
				2*E*alpha*c*(X[j]-x0-c*t)*np.exp(-alpha*(X[j]-x0-c*t)**2)
			#d^2/dt^2 kappa
			kappa_dd=((d_ee-d_gg)/(hbar*omega))*\
				2*E*alpha*c**2*(2*E*alpha*(X[j]-x0-c*t)**2-1)*\
				np.exp(-alpha*(X[j]-x0-c*t)**2)
		else:	
			#kappa
			kappa=((d_ee-d_gg)/(hbar*omega))*(E/np.pi)*\
				(np.arctan(-(X[j]-c*t-x0)*slope)+(np.pi/2))
			#d/dt kappa
			kappa_d=((d_ee-d_gg)/(hbar*omega))*\
				(E*slope*c)/(np.pi*(1+(slope*(X[j]-c*t-x0))**2))
			#d^2/dt^2 kappa		
			kappa_dd=((d_ee-d_gg)/(hbar*omega))*\
				(-2*E*slope**3*c**2*(X[j]-c*t-x0))/\
				(np.pi*(1+(slope*(X[j]-c*t-x0))**2)**2)	

		sour.append(N*((d_ee-d_gg)*sDiff(0,j)+2*np.real(d_ge*\
			(J(1,kappa)*(1j*sDiff(1,j)+sDiff(2,j))+\
			2*kappa_d*0.5*(J(0, kappa)-J(2,kappa))*(1j*Diff(1,j)+Diff(2,j))+\
			(kappa_d**2*0.25*(J(3,kappa)-3*J(1,kappa))+\
			kappa_dd*0.5*(J(0,kappa)-J(2,kappa)))*\
			(1j*rho_now[j][1]+rho_now[j][2])))))			
	return sour
	
def updateMedium(System, init_state, time_prim, x0, x, t, c, delta, gamma,\
		gamma_dec, E, omega, d_eg, d_ee, d_gg, hbar, slope, Om):
	'''
	Function used to solve Bloch equations. 
	Function odeint from scipy.integrate is used here.
	'''
	state_temp = odeint(System, init_state, time_prim, (x0, x, t, c, delta,\
		gamma, gamma_dec, E, omega, d_eg, d_ee, d_gg, hbar, slope, Om))
	return [state_temp[1,0], state_temp[1,1], state_temp[1,2]]	

def bounds(u, left_reflective, right_reflective):	
	'''
	Function used to set boundary conditions. Value True for reflective
	boundaries and False for transmittive.
	'''
	if left_reflective:
		u[0]=0.
	else:
		pass
	if right_reflective:
		u[-1]=0.
	else:
		pass
	return u	
	
def initImpulse(A, x0, c, dx, X, slope=None):
	'''
	Function used to initialize impulse shape and speed.
	'''
	u_now=np.zeros(len(X))
	g=np.zeros(len(X))
	return (u_now, g)		
	
def savePickle(file_name, data):
	'''
	Function used to save results as a picle.
	'''
	with open(file_name, 'wb') as f:
		pickle.dump(data, f)	
	f.close
	return 'Saved.'

def Calculations_t0(A, x0, c, dx, X, slope):
	'''
	Initialization of the impulse and density matrix elements, t=0.
	'''
	u_now, g=initImpulse(A, x0, c, dx, X, slope)
	u_future=u_now
	rho_prev_prev=[[0,0,0] for j in xrange(0, len(X))]
	rho_prev=[[0,0,0] for j in xrange(0, len(X))]
	rho_now=[[0,0,0] for j in xrange(0, len(X))]
	init_state=[[0,0,0] for j in xrange(0, len(X))]	#no population in |e>
	return (u_now, g, u_future, rho_prev_prev, rho_prev, rho_now, init_state)
	
def Calculations_t1(u_now, g, init_state, x0, X, T, c, delta, gamma,\
		gamma_dec, E, omega, d_eg, d_ee, d_gg, hbar, slope, n):
	'''
	Calculations for t=1.
	'''
	u_future=[]
	time_prim=T[0:2]		
	u_future.append(n**2*u_now[1]+(1-n**2)*u_now[0]+(1-n)*dt*g[0])	
	for j in xrange(1, len(X)-1):
		if j>len(X)/4 and j<2*len(X)/4:
			init_state[j]=updateMedium(System, init_state[j], time_prim,\
				x0, X[j], T[1], c, delta, gamma, gamma_dec, E, omega,\
				d_eg, d_ee, d_gg, hbar, slope, u_now[j])				
		u_future.append(0.5*(n**2)*(u_now[j+1]+u_now[j-1])+(1-n**2)*u_now[j]+\
			dt*g[j]-(0.5/eps)*dt**2*v[j])		
	u_future.append(n**2*u_now[len(X)-2]+(1-n**2)*u_now[len(X)-1]+(1-n)*dt*\
		g[len(X)-1])	
	return (u_future, init_state)	
		
def Calculations(i, u_early, u_now, init_state, v, x0, X, T, c, delta, gamma,\
		gamma_dec, E, omega, d_eg, d_ee, d_gg, hbar, slope, n):
	'''
	Calculations for t>1.
	'''
	def temp_f(a, b, output_u):
		u_temp=[]
		try:
			for j in xrange(a, b):
				# active medium is between 0.25 and 0.5 of whole size X
				if j>len(X)/4 and j<2*len(X)/4:
					init_state[j]=updateMedium(System, init_state[j],\
						time_prim, x0, X[j], T[i], c, delta, gamma, gamma_dec,\
						E, omega, d_eg, d_ee, d_gg, hbar, slope, u_now[j])											
				u_temp.append((n**2)*(u_now[j+1]+u_now[j-1])+2*(1-n**2)*\
					u_now[j]-u_early[j]-(1/eps)*dt**2*v[j])
			output_u.put(u_temp)
		except:
			print 'broken'

	
	output_u=Queue()
	u_future=[]
	time_prim=T[i-1:i+1]		
	u_future.append((1/(1+n))*(2*n**2*u_now[1]+2*(1-n**2)*u_now[0]+(n-1)*\
		u_early[0]))
	temp_f(1, len(X)-1, output_u)				
	u_future+=output_u.get()
	u_future.append((1/(1+n))*(2*n**2*u_now[len(X)-2]+2*(1-n**2)*\
		u_now[len(X)-1]+(n-1)*u_early[len(X)-1]))
	return u_future
	
if __name__=="__main__":
	rcParams.update({'font.size': 16})	
	
#################################################
#  1Hz -- 1.52(-16)a.u. (frequency)         	#
#  2piHz -- 0.955(-15)a.u. (angular frequency)  #
#  5V/cm -- 9.73(-10)a.u. (electric field)		#
#  8.5(-30)Cm -- 1a.u. (electric dipole moment)	#
#  5.3cm -- 1.0(9)a.u. (distance) 				#
#  2.42ns -- 1.0(8)a.u. (time)				 	#
#################################################	
	
# atomic units
	hbar=1.
	e=1.
	c=137.
	eps=1/(4*np.pi)	

# CO parameters 
	#~ d_ee=0.07
	#~ d_gg=0
	#~ d_eg=0.63
	#~ d_ge=0.63
	#~ omega=0.3
	#~ gamma_dec=1e-11
	#~ gamma=(omega**3*d_eg**2)/(3*np.pi*eps*hbar*c**3) #5.55e-9
	
# KBr parameters
	#~ d_ee=4.28
	#~ d_gg=0
	#~ d_eg=0.32
	#~ d_ge=0.32
	#~ omega=0.67 #2pi*7(14)Hz (426nm)
	#~ gamma_dec=1e-11 #15us collisions decoherence
	#~ #gamma from Weiskopf-Wigner theorem
	#~ gamma=(omega**3*d_eg**2)/(3*np.pi*eps*hbar*c**3)*0.01 

# NaK parameters
	#~ d_gg=1.1
	#~ d_ee=2.3
	#~ d_eg=1.
	#~ d_ge=2.
	#~ omega=0.1
	#~ gamma_dec=1e-11
	#~ #gamma from Weiskopf-Wigner theorem
	#~ gamma=(omega**3*d_eg**2)/(3*np.pi*eps*hbar*c**3)
	#~ print gamma
	
# representative parameters
	#electric dipole moment matrix elements
	d_gg=0.
	d_ee=1.
	d_eg=1.
	d_ge=1.
	#angular frequency of transition
	omega=0.1
	#colisional decoherence
	gamma_dec=1e-11*0
	#spontaneous emission
	gamma=(omega**3*d_eg**2)/(3*np.pi*eps*hbar*c**3)*0 #W-W theorem
	
	#detuning
	delta=7e-8*0
	#initial impulse amplitude
	A=0.
	#parameter for arctan function 
	#~ x0=-1e11
	#parameter for gaussian function
	x0=-3.4e11
	#drive's amplitude
	E=3.e-7
	#slope of arctan function for drive's envelope
	#~ slope=1.e-10
	#slope of gaussian function for drive's envelope
	slope=3**2*6.5e-23
	#concentration
	N=1e-12

# numerical parameters
	#time step
	dt=1e6
	#space step according to CLF conditions
	dx=c*dt
	# Courant number
	n=c*dt/dx
	#time discretization
	T=np.arange(0, 5e9, dt)
	#space discretization
	X=np.arange(0, 4.e10, dx)
	#boundary conditions	
	left=False
	right=False
	
	print 'gamma: '+str(gamma)
	print 'gamma_dec: ' +str(gamma_dec)
	print 'detuning: ' +str(delta)
	print 'concentration: '+str(N)
	print 'Drive: ' +str(E)
	
	final_impulse=[]
	for i in tqdm(xrange(0, len(T))):
		if i==0:
			u_now, g, u_future, rho_prev_prev, rho_prev, rho_now, init_state=\
				Calculations_t0(A, x0, c, dx, X, slope)
		elif i==1:
			u_future, init_state=Calculations_t1(u_now, g, init_state, x0, X,\
			T, c, delta, gamma, gamma_dec, E, omega, d_eg, d_ee, d_gg, hbar,\
			slope, n)
		else:
			u_future=Calculations(i, u_early, u_now, init_state, v, x0, X, T,\
				c, delta, gamma, gamma_dec, E, omega, d_eg, d_ee, d_gg,\
				hbar, slope, n)

		rho_prev_prev=rho_prev[:]
		rho_prev=rho_now[:]
		rho_now=init_state[:]
		t=T[i]
		v=source(x0, X, t, c, slope, rho_prev_prev, rho_prev, rho_now, dt, N,\
			d_ee, d_gg, d_ge, delta, gamma, gamma_dec, hbar, omega, E)
		u_early=u_now[:]
		u_now=bounds(u_future, left, right)
		final_impulse.append(u_now)
		
	#~ end=[final_impulse[i][-1] for i in xrange(0, len(T))]
	#~ f=plt.figure()
	#~ plt.plot(T, end)
	#~ plt.show()
	#~ savePickle('file_name'+'.pkl', end)
	
	#saving values of a field in each time-space point as a pickle 
	savePickle('filename.pkl', final_impulse)




