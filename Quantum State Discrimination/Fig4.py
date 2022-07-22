# This program is the one used to produce the Fig. 4 in the paper (P_succ vs p for fixed gamma)
# Main program is for NA-LOCCNet (S=2) in the figure (P_succ vs p for fixed gamma)
# P_succ plots for NA-LOCCNet (S=1), LOCCNet are borrowed from supplementary Python programs
# P_succ plots PPT bound (S=1), PPT bound (S=2) are borrowed from supplementary MATLAB program


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
plt.rcParams['axes.ymargin'] = 0.015  # To reset ymargin


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





# permutation funtion A0,A1,B0,B1  to A1,B1,A0,B0    (for 16 X 16 matrix)
def permutation(matrix1):
    prm = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    matrix2 = torch.zeros((16,16), dtype=torch.complex128)
    for i in range(0, 16, 1):  # i runs from 0 to 15 with increments of 1
        for j in range(0, 16, 1):
            matrix2[prm[i],prm[j]] = matrix1[i,j]
    
    return matrix2



def two_qubit_rotation_gate(P,Q,theta):
    Eigen_val, Eigen_vec = torch.linalg.eig(torch.kron(P,Q))

    Evec_0 = Eigen_vec[:,0]
    Evec_1 = Eigen_vec[:,1]
    Evec_2 = Eigen_vec[:,2]
    Evec_3 = Eigen_vec[:,3]


    Ebasis_outer = torch.zeros((4,4,4), dtype =torch.complex128)

    Ebasis_outer[:,:,0] = torch.outer( Evec_0, torch.conj(Evec_0) )
    Ebasis_outer[:,:,1] = torch.outer( Evec_1, torch.conj(Evec_1) )
    Ebasis_outer[:,:,2] = torch.outer( Evec_2, torch.conj(Evec_2) )
    Ebasis_outer[:,:,3] = torch.outer( Evec_3, torch.conj(Evec_3) )


    result_mat = torch.zeros((4,4), dtype =torch.complex128)

    for i in range(0, 4, 1):
        result_mat += torch.exp( (-1.j) * (theta/2) * Eigen_val[i] ) * Ebasis_outer[:,:,i]

    return result_mat









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




#### The followoing 5 functions has to be changed with different rotation gates to find best PQC(Ansatz)

# Two qubit gate at Alice
def U_A(theta_A):

    # temp1 = two_qubit_rotation_gate(X_gate, Y_gate, theta_A)
    temp1 = torch.matmul( CNOT_matrix, two_qubit_rotation_gate(Z_gate, Y_gate, theta_A) )
    temp = torch.kron( temp1 , I_4) # Since we have four qubits we need the kronecker product

    return temp



# Two qubit gate at Bob for Alice measurement output 00
def V_B_00(theta_B_00):

    # temp = two_qubit_rotation_gate(Z_gate, Y_gate, theta_B_00)
    temp = torch.matmul( two_qubit_rotation_gate(Z_gate, Y_gate, theta_B_00), CNOT_matrix )

    return temp



# Two qubit gate at Bob for Alice measurement output 01
def V_B_01(theta_B_01):

    # temp = two_qubit_rotation_gate(Z_gate, Y_gate, theta_B_01)
    temp = torch.matmul( two_qubit_rotation_gate(Z_gate, Y_gate, theta_B_01), CNOT_matrix )

    return temp


# Two qubit gate at Bob for Alice measurement output 10
def V_B_10(theta_B_10):

    # temp = two_qubit_rotation_gate(Z_gate, Y_gate, theta_B_10)
    temp = torch.matmul( two_qubit_rotation_gate(Z_gate, Y_gate, theta_B_10), CNOT_matrix )

    return temp


# Two qubit gate at Bob for Alice measurement output 01
def V_B_11(theta_B_11):

    # temp = two_qubit_rotation_gate(Z_gate, Y_gate, theta_B_11)
    temp = torch.matmul( two_qubit_rotation_gate(Z_gate, Y_gate, theta_B_11), CNOT_matrix )

    return temp

########





def Measuring_first_two_qubits_of_four_qubits(rho,m):    # m is 0,1,2,3(00,01,10,11) corresponding to measurement outcome

    if m==0:    # Measuring 00 at Alice: Probability and post-measurement state
        prob = torch.trace( torch.matmul( torch.kron(projection_00, I_4 ), rho))
        dummy1 = torch.matmul((torch.kron( torch.transpose(x00,0,1) ,I_4)),rho)
        dummy2 = torch.kron(x00,I_4)
        rho_out = (torch.matmul(dummy1,dummy2))/prob
    
    elif m==1:    # Measuring 01 at Alice: Probability and post-measurement state
        prob = torch.trace( torch.matmul( torch.kron(projection_01, I_4 ), rho))
        dummy1 = torch.matmul((torch.kron( torch.transpose(x01,0,1) ,I_4)),rho)
        dummy2 = torch.kron(x01,I_4)
        rho_out = (torch.matmul(dummy1,dummy2))/prob

    elif m==2:    # Measuring 10 at Alice: Probability and post-measurement state
        prob = torch.trace( torch.matmul( torch.kron(projection_10, I_4 ), rho))
        dummy1 = torch.matmul((torch.kron( torch.transpose(x10,0,1) ,I_4)),rho)
        dummy2 = torch.kron(x10,I_4)
        rho_out = (torch.matmul(dummy1,dummy2))/prob    

    elif m==3:    # Measuring 11 at Alice: Probability and post-measurement state
        prob = torch.trace( torch.matmul( torch.kron(projection_11, I_4 ), rho))
        dummy1 = torch.matmul((torch.kron( torch.transpose(x11,0,1) ,I_4)),rho)
        dummy2 = torch.kron(x11,I_4)
        rho_out = (torch.matmul(dummy1,dummy2))/prob

    return rho_out, prob
 



# Input: quantum state as density and gate. Output: quantum state as density
# This function accepts any order matrices (which are compatible for multiplication)
def density_output_of_Qgate(rho,Qgate):   
    rho_out = torch.matmul( torch.matmul(Qgate,rho), dagger(Qgate) )

    return rho_out


# Function for computing dagger (conjugate transpose) of a matrix(vector)
def dagger(vector):
    temp = torch.transpose(torch.conj(vector),0,1)

    return temp




# Function for computing success probabilities when input is rho_1. 
# Here b=0 and b=1 are used to find success probability for rho_0 and rho_1 respectively
# Here b=1 and b=0 are used to find failure probability for rho_0 and rho_1 respectively

def circuit(rho_in, theta_A, theta_B_00, theta_B_01, theta_B_10, theta_B_11, b, p):

    rho_input = permutation( torch.kron( rho_in,rho_in ) )

    rho_1 = density_output_of_Qgate(rho_input, U_A( theta_A ) )

    # Measuring 00 at Alice: Probability and post-measurement state
    rho_2_00, P_Alice_00 = Measuring_first_two_qubits_of_four_qubits(rho_1, 0)
    
    # Measuring 01 at Alice: Probability and post-measurement state
    rho_2_01, P_Alice_01 = Measuring_first_two_qubits_of_four_qubits(rho_1, 1)

    # Measuring 10 at Alice: Probability and post-measurement state
    rho_2_10, P_Alice_10 = Measuring_first_two_qubits_of_four_qubits(rho_1, 2)

    # Measuring 11 at Alice: Probability and post-measurement state
    rho_2_11, P_Alice_11 = Measuring_first_two_qubits_of_four_qubits(rho_1, 3)


    # Communicating through noisy channels with bit flip probability p
    # probability of observing 00,01,10,11 and corresponding postmeasurement state has to be found
    # and this is will be used for further quantum processing at Bob

    # Receiving 00 at Bob: Probability and post-measurement state
    cap_P_Alice_00 = ((1-p)**2) * P_Alice_00 + ((1-p)*p) * P_Alice_01 + (p*(1-p)) * P_Alice_10 + (p**2) * P_Alice_11
    cap_rho_2_00 = ((1-p)**2) * rho_2_00 + ((1-p)*p) * rho_2_01 + (p*(1-p)) * rho_2_10 + (p**2) * rho_2_11

    # Receiving 01 at Bob: Probability and post-measurement state
    cap_P_Alice_01 = ((1-p)*p) * P_Alice_00 + ((1-p)**2) * P_Alice_01 + (p**2) * P_Alice_10 + (p*(1-p)) * P_Alice_11
    cap_rho_2_01 = ((1-p)*p) * rho_2_00 + ((1-p)**2) * rho_2_01 + (p**2) * rho_2_10 + (p**(1-p)) * rho_2_11

    # Receiving 10 at Bob: Probability and post-measurement state
    cap_P_Alice_10 = (p*(1-p)) * P_Alice_00 + (p**2) * P_Alice_01 + ((1-p)**2) * P_Alice_10 + ((1-p)*p) * P_Alice_11
    cap_rho_2_10 = (p*(1-p)) * rho_2_00 + (p**2) * rho_2_01 + ((1-p)**2) * rho_2_10 + ((1-p)*p) * rho_2_11

    # Receiving 11 at Bob: Probability and post-measurement state
    cap_P_Alice_11 = (p**2) * P_Alice_00 + (p*(1-p)) * P_Alice_01 + ((1-p)*p) * P_Alice_10 + ((1-p)**2) * P_Alice_11
    cap_rho_2_11 = (p**2) * rho_2_00 + (p*(1-p)) * rho_2_01 + ((1-p)*p) * rho_2_10 + ((1-p)**2) * rho_2_11


    # Output of gate at Bob corresponding to four different observations of nosiy channel outputs (00,01,10,11) 
    rho_3_00 = cap_P_Alice_00 * density_output_of_Qgate(cap_rho_2_00,V_B_00(theta_B_00))
    rho_3_01 = cap_P_Alice_01 * density_output_of_Qgate(cap_rho_2_01,V_B_01(theta_B_01))
    rho_3_10 = cap_P_Alice_10 * density_output_of_Qgate(cap_rho_2_10,V_B_10(theta_B_10))
    rho_3_11 = cap_P_Alice_11 * density_output_of_Qgate(cap_rho_2_11,V_B_11(theta_B_11))

    # Post-processed state of Bob (before measurement)
    rho_3 = rho_3_00 + rho_3_01 + rho_3_10 + rho_3_11


    # Parity projective Measurement at Bob

    if b==0:
        temp = torch.trace( torch.matmul( (projection_00 + projection_11), rho_3))
        
    elif b==1:
        temp = torch.trace( torch.matmul( (projection_01 + projection_10), rho_3))
        

    return temp






######  END OF FUNCTIONS




# Defining 4*4 CNOT matrix (first qubit as control and second qubit as target)
CNOT_matrix = torch.zeros((4,4), dtype=torch.complex128)
CNOT_matrix[0,0] = 1
CNOT_matrix[1,1] = 1
CNOT_matrix[2,3] = 1
CNOT_matrix[3,2] = 1


# Defining 4*4 reverse CNOT matrix (second qubit as control and first qubit as target)
CNOT_matrix_reverse = torch.zeros((4,4), dtype=torch.complex128)
CNOT_matrix_reverse[0,0] = 1
CNOT_matrix_reverse[1,3] = 1
CNOT_matrix_reverse[2,2] = 1
CNOT_matrix_reverse[3,1] = 1




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
I_4=torch.eye(4, dtype =torch.complex128)

pi=torch.tensor(math.pi, dtype=torch.complex128)


X_gate = torch.zeros((2,2), dtype =torch.complex128)
Y_gate = torch.zeros((2,2), dtype =torch.complex128)
Z_gate = torch.zeros((2,2), dtype =torch.complex128)

X_gate[0,1] = 1
X_gate[1,0] = 1

Y_gate[0,1] = -1.j
Y_gate[1,0] = 1.j

Z_gate[0,0] = 1
Z_gate[1,1] = -1



x00=torch.zeros((4,1),dtype=torch.complex128)
x00[0,0]=1

x01=torch.zeros((4,1),dtype=torch.complex128)
x01[1,0]=1

x10=torch.zeros((4,1),dtype=torch.complex128)
x10[2,0]=1

x11=torch.zeros((4,1),dtype=torch.complex128)
x11[3,0]=1

projection_00 = torch.matmul( x00, dagger(x00)) # Two qubit projection matrix with output 00
projection_01 = torch.matmul( x01, dagger(x01)) # Two qubit projection matrix with output 01
projection_10 = torch.matmul( x10, dagger(x10)) # Two qubit projection matrix with output 10
projection_11 = torch.matmul( x11, dagger(x11)) # Two qubit projection matrix with output 11





######################################


# For rho_phi_plus input state 
rho_in_0 = rho_phi_plus    


avg_P_succ=torch.zeros((6),dtype=torch.complex128)
# avg_P_succ1=torch.zeros((11),dtype=torch.complex128)


    
gamma = torch.tensor(0.8, dtype=torch.complex128)  # represents prob of noisy channels between Alice and Bob (0 to 0.5 with increments of 0.1)
rho_in_1 = two_qubit_Amplitude_damping_channel_with_phi_minus_input(gamma)


for i in range(0, 6, 1):   # i runs from 0 to 5 with increments of 1
    
    p = torch.tensor(i/10, dtype=torch.complex128)


    def cost(theta_A, theta_B_00, theta_B_01, theta_B_10, theta_B_11):
        P_fail_0 = circuit(rho_in_0, theta_A, theta_B_00, theta_B_01, theta_B_10, theta_B_11, 1, p)

        P_fail_1 = circuit(rho_in_1, theta_A, theta_B_00, theta_B_01, theta_B_10, theta_B_11, 0, p)

        Prob_fail = P_fail_0 + P_fail_1

        return Prob_fail


    theta_A = torch.tensor(6.1, requires_grad=True)   # angle in radians for gate at Alice
    theta_B_00 = torch.tensor(6.1, requires_grad=True)   # angle in radians for gate at Bob if Alice measurement is 00
    theta_B_01 = torch.tensor(6.1, requires_grad=True)   # angle in radians for gate at Bob if Alice measurement is 01
    theta_B_10 = torch.tensor(6.1, requires_grad=True)   # angle in radians for gate at Bob if Alice measurement is 10
    theta_B_11 = torch.tensor(6.1, requires_grad=True)   # angle in radians for gate at Bob if Alice measurement is 11

    opt = torch.optim.Adam([ theta_A, theta_B_00, theta_B_01, theta_B_10, theta_B_11 ], lr = 0.01)   # lr represents learning rate

    steps = 1000

    for itr in range(steps):
        opt.zero_grad()
        loss = cost(theta_A, theta_B_00, theta_B_01, theta_B_10, theta_B_11)
        loss.backward()
        opt.step()
        # if itr % 300 ==0:  #print for every 200 steps
        #     print('itr', itr, 'loss', loss,'theta_0',theta_0,'theta_1',theta_1,'theta_2',theta_2)

    # print('p',p,'gamma',gamma)
    # print('theta_A',theta_A,'theta_B_00',theta_B_00,'theta_B_01',theta_B_01,'theta_B_10',theta_B_10,'theta_B_11',theta_B_11)

    P_succ_0 = circuit(rho_in_0, theta_A, theta_B_00, theta_B_01, theta_B_10, theta_B_11, 0, p)

    P_succ_1 = circuit(rho_in_1, theta_A, theta_B_00, theta_B_01, theta_B_10, theta_B_11, 1, p)

    avg_P_succ[i] = (P_succ_0 + P_succ_1)/2
    

print(avg_P_succ)


# The following by the fact that Bob can always add classical noise to increase the avg_P_succ
# In avg_P_succ for p=0.3(i=3) and p=0.4(i=4) the avg_P_succ is less than the corresponding value at p=0.5
# So in these cases Bob adds classical noise to make it p=0.5 (i.e., completely discards the measurement output of Alice)
# By this technique we can further enhance the avg_P_succ at p=0.3 and p=0.4
for i in range(0, 6, 1):
    if avg_P_succ[i].real <= avg_P_succ[5].real:
        avg_P_succ[i] = avg_P_succ[5]



#Plots
xpoints = torch.arange(0, 0.6, 0.1)  # X-axis coordinates (correspondng to probability of bit flip p)

avg_P_succ = avg_P_succ.detach().numpy()

plt.plot(xpoints, avg_P_succ, 'b-') #for two pairs



# sinlge pair with training for different p and gamma
# following values are from the previous programme for single pair with gamma = 0.8
avg_P_succ_single_pair_NA_LOCCNet_gamma_0_point_8 = [0.8605, 0.8124, 0.7691, 0.7332, 0.7088, 0.6998]


# following values are from the previous programme for single pair with gamma = 0.3
# avg_P_succ_single_pair_NA_LOCCNet_gamma_0_point_3 = [0.9315, 0.8482, 0.7658, 0.6858, 0.6133, 0.5750]


plt.plot(xpoints, avg_P_succ_single_pair_NA_LOCCNet_gamma_0_point_8, 'r-')
# plt.plot(xpoints, avg_P_succ_single_pair_NA_LOCCNet_gamma_0_point_3, 'r-')






# sinlge pair without training. 
# using same rotation angles as that of LOCCNet(i.e., trained at p=0)
# following values are from the previous programme for single pair with gamma = 0.8
avg_P_succ_single_pair_LOCCNet_gamma_0_point_8 = [0.8606, 0.8106, 0.7607, 0.7108, 0.6609, 0.6109]


# following values are from the previous programme for single pair with gamma = 0.3
# avg_P_succ_single_pair_LOCCNet_gamma_0_point_3 = [0.9316, 0.8479, 0.7642, 0.6804, 0.5967, 0.5130]


plt.plot(xpoints, avg_P_succ_single_pair_LOCCNet_gamma_0_point_8, 'r--*')
# plt.plot(xpoints, avg_P_succ_single_pair_LOCCNet_gamma_0_point_3, 'r--*')






## PPT and POVM bounds are obtained from MATLAB for single pair (these 10 values are for gamma 0 to 1.0 with increments of 0.1)
# avg_P_succ_PPT_single_pair = [1.0000, 0.9757, 0.9530, 0.9325, 0.9144, 0.8991, 0.8865, 0.8764, 0.8680, 0.8607, 0.8536]

# avg_P_succ_POVM_single_pair = [1.0000, 0.9982, 0.9928, 0.9841, 0.9723, 0.9578, 0.9408, 0.9216, 0.9006, 0.8778, 0.8536]

avg_P_succ_PPT_single_pair_gamma_0_point_8 = [0.8680, 0.8680, 0.8680, 0.8680, 0.8680, 0.8680]
# avg_P_succ_PPT_single_pair_gamma_0_point_3 = [0.9325, 0.9325, 0.9325, 0.9325, 0.9325, 0.9325]


# PPT and POVM bounds are obtained from MATLAB for two pair (these 10 values are for gamma 0 to 1.0 with increments of 0.1)
# avg_P_succ_PPT_two_pair = [1.0000, 0.9988, 0.9951, 0.9893, 0.9818, 0.9731, 0.9639, 0.9550, 0.9469, 0.9397, 0.9330]

# avg_P_succ_POVM_two_pair = [1.0000, 1.0000, 0.9998, 0.9992, 0.9975, 0.9941, 0.9885, 0.9799, 0.9679, 0.9523, 0.9330]


avg_P_succ_PPT_two_pair_gamma_0_point_8 = [0.9469, 0.9469, 0.9469, 0.9469, 0.9469, 0.9469]
# avg_P_succ_PPT_two_pair_gamma_0_point_3 = [0.9893, 0.9893, 0.9893, 0.9893, 0.9893, 0.9893]


    
# # PPT Bound for gamma 0.8
plt.plot(xpoints, avg_P_succ_PPT_two_pair_gamma_0_point_8, 'b:o')
plt.plot(xpoints, avg_P_succ_PPT_single_pair_gamma_0_point_8, 'r:o')



# # PPT Bound for gamma 0.3
# plt.plot(xpoints, avg_P_succ_PPT_two_pair_gamma_0_point_3, 'b:o')
# plt.plot(xpoints, avg_P_succ_PPT_single_pair_gamma_0_point_3, 'r:o')










# plt.legend(["NA-LOCCNet gamma=0.3 with two pairs", "LOCCNet gamma=0.3 with single pair (trained via (eq))", "LOCCNet gamma=0.3 with single pair", "PPT bound-two pairs", "PPT bound-single pair"], loc="lower left", fontsize=12)

# plt.xlabel("Probability of bit flip p", fontsize=13)
# plt.ylabel("Average success proability", fontsize=13)


plt.grid(True)
plt.tight_layout()
plt.show()