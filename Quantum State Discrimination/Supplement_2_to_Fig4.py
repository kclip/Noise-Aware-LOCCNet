# This is for NA-LOCCNet (S=1) in the figure (P_succ vs p for fixed gamma)

# Python (Pytorch) code to implement Quantum state discrimination in the LOCCNet paper with optimization
# with noisy communication
# for one pair of input state

# Here we are using the same ansatz as in LOCCNet paper
# For gamma=0.8 we optimize the rotation angles at Alice and Bob

# In the plot X-axis is bit flip probability (p), Y-axis is average success probability


from unittest import FunctionTestCase
import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
from torch import linalg


plt.rcParams["font.family"] = "Times New Roman"  # To set global font for plots
plt.rcParams['axes.xmargin'] = 0  # To reset xmargin
plt.rcParams['axes.ymargin'] = 0  # To reset ymargin


################# FUNCTIONS

# Rx(theta) gate
def R_xgate(theta):
    R_xgate1 = torch.zeros((2,2), dtype=torch.complex128 )
    R_xgate1[0,0] = torch.cos(theta/2)
    R_xgate1[0,1] = -(1.j)*torch.sin(theta/2)
    R_xgate1[1,0] = -(1.j)*torch.sin(theta/2)
    R_xgate1[1,1] = torch.cos(theta/2)

    return R_xgate1


# Ry(theta) gate
def R_ygate(theta):
    R_ygate1 = torch.zeros((2,2), dtype=torch.complex128 )
    R_ygate1[0,0] = torch.cos(theta/2)
    R_ygate1[0,1] = -torch.sin(theta/2)
    R_ygate1[1,0] = torch.sin(theta/2)
    R_ygate1[1,1] = torch.cos(theta/2)

    return R_ygate1



# Rz(theta) gate
def R_zgate(theta):
    R_zgate1 = torch.zeros((2,2), dtype=torch.complex128 )
    R_zgate1[0,0] = torch.exp( (-1.j) * (theta/2) )
    R_zgate1[1,1] = torch.exp( (1.j) * (theta/2) )

    return R_zgate1



# Defining TWO-QUBIT amplitude damping channel for |phi^-> input eq (S56) in page 18 of LOCCNet paper
def two_qubit_Amplitude_damping_channel_with_phi_minus_input(gamma):
    rho = torch.zeros((4,4),dtype=torch.complex128)
    rho[0,0] = (1+gamma**2)/2      
    rho[0,3] = (gamma-1)/2
    rho[3,0] = (gamma-1)/2
    rho[3,3] = ((1-gamma)**2)/2      
    rho[1,1] = (gamma-gamma**2)/2
    rho[2,2] = (gamma-gamma**2)/2

    return rho




#### The followoing 3 functions has to be changed with different rotation gates to find best PQC(Ansatz)

# One qubit gate at Alice
def U_A(theta_A):
    temp = torch.kron( R_ygate(theta_A) , I_2) # Since we have two qubits we need the kronecker product

    return temp



# One qubit gate at Bob for Alice measurement output 0
def V_B_0(theta_B_0):
    temp = R_ygate(theta_B_0)

    return temp



# One qubit gate at Bob for Alice measurement output 1
def V_B_1(theta_B_1):
    temp = R_ygate(theta_B_1)

    return temp

####


def Measuring_first_qubit_of_two_qubits(rho,m):    # m is 0 or 1 corresponding to measurement outcome

    if m==0:    # Measuring 0 at Alice: Probability and post-measurement state
        prob = torch.trace( torch.matmul( torch.kron(projection_0, I_2 ), rho))
        dummy1 = torch.matmul((torch.kron( torch.transpose(x0,0,1) ,I_2)),rho)
        dummy2 = torch.kron(x0,I_2)
        rho_out = (torch.matmul(dummy1,dummy2))/prob
    
    elif m==1:  # Measuring 1 at Alice: Probability and post-measurement state
        prob = torch.trace( torch.matmul( torch.kron(projection_1, I_2 ), rho))
        dummy1 = torch.matmul((torch.kron( torch.transpose(x1,0,1) ,I_2)),rho)
        dummy2 = torch.kron(x1,I_2)
        rho_out = (torch.matmul(dummy1,dummy2))/prob

    return rho_out, prob
 

# Input: quantum state as density and gate. Output: quantum state as density
# This function accepts any order matrices (which are compatible for multiplication)
def density_output_of_Qgate(rho,Qgate):   
    rho_out = torch.matmul( torch.matmul(Qgate,rho), dagger(Qgate) )

    return rho_out

def dagger(vector):
    temp = torch.transpose(torch.conj(vector),0,1)

    return temp



# Function for computing success probabilities when input is rho_1. 
# Here b=0 and b=1 are used to find success probability for rho_0 and rho_1 respectively
# Here b=1 and b=0 are used to find failure probability for rho_0 and rho_1 respectively



def circuit(rho_in,theta_A,theta_B_0,theta_B_1,b,p):

    rho_1 = density_output_of_Qgate(rho_in, U_A( theta_A ) )

    # Measuring 0 at Alice: Probability and post-measurement state
    rho_2_0, P_Alice_0 = Measuring_first_qubit_of_two_qubits(rho_1, 0)
    
    # Measuring 1 at Alice: Probability and post-measurement state
    rho_2_1, P_Alice_1 = Measuring_first_qubit_of_two_qubits(rho_1, 1)


    # Communicating through noisy channel with bit flip probability p
    # probability of observing 0 or 1 and corresponding postmeasurement state has to be found
    # and this is will be used for further quantum processing at Bob

    # Receiving 0 at Bob: Probability and post-measurement state
    cap_P_Alice_0 = (1-p) * P_Alice_0 + p * P_Alice_1
    cap_rho_2_0 = (1-p) * rho_2_0 + p * rho_2_1

    # Receiving 1 at Bob: Probability and post-measurement state
    cap_P_Alice_1 = p * P_Alice_0 + (1-p) * P_Alice_1
    cap_rho_2_1 = p * rho_2_0 + (1-p) * rho_2_1



    # Output of R_Y gate at Bob corresponding to two different observations of nosiy channel outputs (0 and 1) 
    rho_3_0 = cap_P_Alice_0 * density_output_of_Qgate(cap_rho_2_0,V_B_0(theta_B_0))
    rho_3_1 = cap_P_Alice_1 * density_output_of_Qgate(cap_rho_2_1,V_B_1(theta_B_1))

    # Post-processed state of Bob (before measurement)
    rho_3 = rho_3_0 + rho_3_1


    # Measurement at Bob

    if b==0:
        temp = torch.trace( torch.matmul( projection_0, rho_3))
        
    elif b==1:
        temp = torch.trace( torch.matmul( projection_1, rho_3))
        

    return temp







######  END OF FUNCTIONS




rho_phi_plus = torch.zeros((4,4),dtype=torch.complex128)
rho_phi_plus[0,0] = 0.5
rho_phi_plus[0,3] = 0.5
rho_phi_plus[3,0] = 0.5
rho_phi_plus[3,3] = 0.5


rho_phi_minus = torch.zeros((4,4),dtype=torch.complex128)
rho_phi_minus[0,0] = 0.5
rho_phi_minus[0,3] = -0.5
rho_phi_minus[3,0] = 0.5
rho_phi_minus[3,3] = -0.5


I_2=torch.eye(2, dtype =torch.complex128)

pi=torch.tensor(math.pi, dtype=torch.complex128)


x0=torch.zeros((2,1),dtype=torch.complex128)
x0[0,0]=1

x1=torch.zeros((2,1),dtype=torch.complex128)
x1[1,0]=1

projection_0 = torch.matmul( x0, dagger(x0) ) # single qubit projection matrix with output 0
projection_1 = torch.matmul( x1, dagger(x1) ) # single qubit projection matrix with output 1





######################################


# For rho_phi_plus input state 
rho_in_0 = rho_phi_plus    


avg_P_succ=torch.zeros((6),dtype=torch.complex128)
# avg_P_succ1=torch.zeros((11),dtype=torch.complex128)


    
gamma = torch.tensor(0.8, dtype=torch.complex128)  # represents amplitude damping noise
rho_in_1 = two_qubit_Amplitude_damping_channel_with_phi_minus_input(gamma)

for i in range(0, 6, 1):   # i runs from 0 to 5 with increments of 1
    
    p = torch.tensor(i/10, dtype=torch.complex128) #prob of noisy channels between Alice and Bob (0 to 0.5 with increments of 0.1)


    def cost(theta_A, theta_B_0, theta_B_1):
        P_fail_0 = circuit(rho_in_0, theta_A, theta_B_0, theta_B_1, 1, p)

        P_fail_1 = circuit(rho_in_1, theta_A, theta_B_0, theta_B_1, 0, p)

        Prob_fail = P_fail_0 + P_fail_1

        return Prob_fail


    theta_A = torch.tensor(6.1, requires_grad=True)   # angle in radians for gate at Alice
    theta_B_0 = torch.tensor(6.1, requires_grad=True)   # angle in radians for gate at Bob if Alice measurement is 0
    theta_B_1 = torch.tensor(6.1, requires_grad=True)   # angle in radians for gate at Bob if Alice measurement is 1

    opt = torch.optim.Adam([ theta_A, theta_B_0, theta_B_1 ], lr = 0.01)   # lr represents learning rate

    steps = 1000

    for itr in range(steps):
        opt.zero_grad()
        loss = cost(theta_A, theta_B_0, theta_B_1)
        loss.backward()
        opt.step()
        # if itr % 300 ==0:  #print for every 200 steps
        #     print('itr', itr, 'loss', loss,'theta_0',theta_0,'theta_1',theta_1,'theta_2',theta_2)

    # print('p',p,'gamma',gamma)
    # print('theta_A',theta_A,'theta_B_0',theta_B_0,'theta_B_1',theta_B_1)

    P_succ_0 = circuit(rho_in_0, theta_A, theta_B_0, theta_B_1, 0, p)

    P_succ_1 = circuit(rho_in_1, theta_A, theta_B_0, theta_B_1, 1, p)

    avg_P_succ[i] = (P_succ_0 + P_succ_1)/2
    






#Plots
xpoints = torch.arange(0, 0.6, 0.1)  # X-axis coordinates (correspondng to probability of bit flip p)

avg_P_succ = avg_P_succ.detach().numpy()

print(avg_P_succ)  # Use these values to plot

plt.plot(xpoints, avg_P_succ, 'r--')





# plt.plot(xpoints, avg_P_succ1, 'g:')

# PPT and POVM bounds are obtained from MATLAB
# avg_P_succ_PPT = [1.0000, 0.9757, 0.9530, 0.9325, 0.9144, 0.8991, 0.8865, 0.8764, 0.8680, 0.8607, 0.8536]

# avg_P_succ_POVM = [1.0000, 0.9982, 0.9928, 0.9841, 0.9723, 0.9578, 0.9408, 0.9216, 0.9006, 0.8778, 0.8536]


# plt.plot(xpoints, avg_P_succ_PPT, 'b-')
# plt.plot(xpoints, avg_P_succ_POVM, 'y-')



plt.legend(["LOCCNet trained at gamma=0.8"], loc="lower left", fontsize=12)
# plt.legend(["LOCCNet","Circuit designed for noiseless case","PPT Bound", "POVM Bound using SDP", "POVM Bound using Helstrom"], loc="lower left", fontsize=12)
plt.xlabel("Probability of bit flip p", fontsize=13)
plt.ylabel("Average success proability", fontsize=13)


plt.grid(True)
plt.tight_layout()
plt.show()


