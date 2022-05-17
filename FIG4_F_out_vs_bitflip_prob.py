# Python (Pytorch) code to implement NA-LOCCNet and plot average output fidelity vs bit flip probability for fixed input fidelity.

from unittest import FunctionTestCase
import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np


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



# Defining S-state   \rho = F_in|\phi^+ ><\phi^+| + (1-F_in) |00><00|
def rho_Sstate(F_in):
    rho = torch.zeros((4,4),dtype=torch.complex128)
    rho[0,0] = 1-F_in/2      #For S-state
    rho[0,3] = F_in/2
    rho[3,0] = F_in/2
    rho[3,3] = F_in/2

    return rho


# Defining state with single quantum bitflip    \rho = F_in|\phi^+ ><\phi^+| + (1-F_in) |\psi^+><\psi^+|
def rho_single_quantum_bitflip_state(F_in):
    rho = torch.zeros((4,4),dtype=torch.complex128)
    rho[0,0] = F_in/2
    rho[0,3] = F_in/2
    rho[3,0] = F_in/2
    rho[3,3] = F_in/2

    rho[1,1] = (1-F_in)/2
    rho[1,2] = (1-F_in)/2
    rho[2,1] = (1-F_in)/2
    rho[2,2] = (1-F_in)/2

    return rho


# Finding q_ij and qF_ij (= q_ij * F_ij) for all i,j \in \{0,1\}
def q_ij_and_qF_ij(x_ij, rho_output):
    dummy1=torch.matmul((torch.kron( torch.transpose(x_ij,0,1) ,I_4)),rho_output)
    dummy2=torch.kron(x_ij,I_4)
    rho_ij=torch.matmul(dummy1,dummy2)
    q_ij=torch.trace(rho_ij)
    # rho_00=torch.divide(rho_00,q_00)
    qF_ij=torch.trace(torch.mm(rho_ij,phi_matrix))  # represents q_ij*F_ij for all i,j \in \{0,1\}

    return q_ij, qF_ij


# Decision rule 1: Accepting 00/11 and rejecting 01/10
def decisionrule1(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11):
    prob_succ = ((p**2 +(1-p)**2)*(q_00 + q_11) + 2*p*(1-p)*(q_01 + q_10))
    F_avg = torch.divide(((p**2 +(1-p)**2)*(qF_00 + qF_11) + 2*p*(1-p)*(qF_01 + qF_10)), prob_succ)

    return F_avg, prob_succ


# Decision rule 2: Accepting 00 and rejecting 01/10/11
def decisionrule2(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11):
    prob_succ = ((1-p)**2 * q_00) +(p**2 * q_11) + (p*(1-p)*q_10) + ((1-p)*p*q_01)
    F_avg = torch.divide( ((1-p)**2 * qF_00) +(p**2 * qF_11) + (p*(1-p)*qF_10) + ((1-p)*p*qF_01), prob_succ)

    return F_avg, prob_succ




# Decision rule 11: Accepting 00/11 and rejecting 01/10
def decisionrule11(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11):
    prob_succ = ((p**2 +(1-p)**2)*(q_00 + q_11) + 2*p*(1-p)*(q_01 + q_10))
    F_avg = torch.divide(((p**2 +(1-p)**2)*(qF_00 + qF_11) + 2*p*(1-p)*(qF_01 + qF_10)), prob_succ)

    return F_avg


# Decision rule 22: Accepting 00 and rejecting 01/10/11
def decisionrule22(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11):
    prob_succ = ((1-p)**2 * q_00) +(p**2 * q_11) + (p*(1-p)*q_10) + ((1-p)*p*q_01)
    F_avg = torch.divide( ((1-p)**2 * qF_00) +(p**2 * qF_11) + (p*(1-p)*qF_10) + ((1-p)*p*qF_01), prob_succ)

    return F_avg



# permutation funtion A0,A1,B0,B1  to A1,B1,A0,B0
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






###################  END OF FUNCTIONS


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


I_2=torch.eye(2, dtype =torch.complex128)
I_4=torch.eye(4, dtype =torch.complex128)



# pure entangled state (ebit)
phi_matrix=torch.zeros((4,4),dtype=torch.complex128)
phi_matrix[0,0]=0.5
phi_matrix[0,3]=0.5
phi_matrix[3,0]=0.5
phi_matrix[3,3]=0.5


x0=torch.zeros((4,1),dtype=torch.complex128)
x0[0,0]=1

x1=torch.zeros((4,1),dtype=torch.complex128)
x1[1,0]=1

x2=torch.zeros((4,1),dtype=torch.complex128)
x2[2,0]=1

x3=torch.zeros((4,1),dtype=torch.complex128)
x3[3,0]=1


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







xpoints = torch.arange(0, 0.55, 0.05)  # X-axis coordinates (corresponds to probability from 0 to 0.5 with 0.05 intervals. 11 points)



qF_00=torch.zeros((11),dtype=torch.complex128) # represents the product of q_00 and F_00. This is to avoid NaN case if q_00=0
qF_01=torch.zeros((11),dtype=torch.complex128)
qF_10=torch.zeros((11),dtype=torch.complex128)
qF_11=torch.zeros((11),dtype=torch.complex128)

q_00=torch.zeros((11),dtype=torch.complex128)
q_01=torch.zeros((11),dtype=torch.complex128)
q_10=torch.zeros((11),dtype=torch.complex128)
q_11=torch.zeros((11),dtype=torch.complex128)


F_opt=torch.zeros((11),dtype=torch.complex128)
P_opt=torch.zeros((11),dtype=torch.complex128)


# p_training = torch.tensor(0.1, dtype = torch.complex128)  # probability of bit-flip error of classical channel (for training)
F_in = torch.tensor(0.6, dtype = torch.complex128) # Input fidelity (constant) 


figure = 3 # to get average fidelity plots
# figure = 4 # to get average success probability plots


########################################

















# ##############  DEJMPS circuit


rho = rho_Sstate(F_in)      #For S-state

# rho = rho_single_quantum_bitflip_state(F_in)  #For single quantum bit flip to ebit state


rho_in = torch.kron(rho,rho)
rho_output=torch.zeros((16,16),dtype=torch.complex128)

R_x1 =  R_xgate(pi/2)   # R_x(+pi/2) gate
R_x2 =  R_xgate(-pi/2)   # R_x(+pi/2) gate


U_1 = torch.kron(torch.kron(R_x1,R_x1),torch.kron(R_x2,R_x2))
U_2 = torch.kron(CNOT_matrix,CNOT_matrix)

U = torch.matmul( U_2, U_1 )  # circuit of DEJMPS

V = permutation(U)   # Permutation A0,A1,B0,B1  to A1,B1,A0,B0

rho_output = torch.matmul( torch.matmul(V,rho_in), torch.transpose(torch.conj(V),0,1) )


q_00, qF_00 = q_ij_and_qF_ij(x0, rho_output)
q_01, qF_01 = q_ij_and_qF_ij(x1, rho_output)
q_10, qF_10 = q_ij_and_qF_ij(x2, rho_output)
q_11, qF_11 = q_ij_and_qF_ij(x3, rho_output)



for prob in range(0, 11, 1):   # i runs from 0 to 5

    p = torch.tensor(0 + (prob*0.05), dtype = torch.complex128)  # probability of bit-flip error of classical channels


    ###########################  DECISION RULES

    F_avg, P_success = decisionrule1(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00/11 and rejecting 01/10

    # F_avg, P_success = decisionrule2(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00 and rejecting 01/10/11


    ############################

    F_opt[prob] = F_avg
    P_opt[prob] = P_success


if figure ==3:
    plt.plot(xpoints, F_opt, 'g|-')

if figure ==4:
    plt.plot(xpoints, P_opt, 'g|-')


###################### END of DEJMPS circuit













##############  LOCCNet circuit, trained at p=0 and fixed F_in using theoretical theta from paper


rho = rho_Sstate(F_in)      #For S-state

# rho = rho_single_quantum_bitflip_state(F_in)  #For single quantum bit flip to ebit state


rho_in = torch.kron(rho,rho)
rho_output=torch.zeros((16,16),dtype=torch.complex128)


U_1 = torch.kron(CNOT_matrix_reverse,CNOT_matrix_reverse)    # First two reverse CNOT's at Alice and Bob in LOCCNet
U_2 = torch.kron(I_4,CNOT_matrix)   #  CNOT at Bob in LOCCNet

theta_theoretical = torch.acos(torch.tensor(1, dtype = torch.complex128) - F_in) + pi  # GIVEN IN LOCCNet paper

R_y =  R_ygate(theta_theoretical)   # R_y(theta) gate

U_rotation = torch.kron(torch.kron(I_2,R_y),torch.kron(I_2,R_y))


U = torch.matmul( torch.matmul( U_rotation, U_2), U_1)  # circuit of LOCCNet

V = permutation(U)      # Permutation A0,A1,B0,B1  to A1,B1,A0,B0

rho_output = torch.matmul( torch.matmul(V,rho_in), torch.transpose(torch.conj(V),0,1) )


q_00, qF_00 = q_ij_and_qF_ij(x0, rho_output)
q_01, qF_01 = q_ij_and_qF_ij(x1, rho_output)
q_10, qF_10 = q_ij_and_qF_ij(x2, rho_output)
q_11, qF_11 = q_ij_and_qF_ij(x3, rho_output)



for prob in range(0, 11, 1):   # prob runs from 0 to 10 with increments of 1

    p = torch.tensor(0 + (prob*0.05), dtype = torch.complex128)  # probability of bit-flip error of classical channels


    ###########################  DECISION RULES

    # F_avg, P_success = decisionrule1(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00/11 and rejecting 01/10

    F_avg, P_success = decisionrule2(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00 and rejecting 01/10/11


    ############################

    F_opt[prob] = F_avg
    P_opt[prob] = P_success


if figure ==3:
    plt.plot(xpoints, F_opt, 'mo:')

if figure ==4:
    plt.plot(xpoints, P_opt, 'mo:')


###################### END OF LOCCNet circuit, trained at p=0 and fixed F_in using theoretical theta from paper














#################################################################
###################  LOCCNet circuit, trained at various p and fixed F_in



rho = rho_Sstate(F_in)      #For S-state

# rho = rho_single_quantum_bitflip_state(F_in)  #For single quantum bit flip to ebit state


rho_in = torch.kron(rho,rho)
rho_output=torch.zeros((16,16),dtype=torch.complex128)


U_1 = torch.kron(CNOT_matrix_reverse,CNOT_matrix_reverse)    # First two reverse CNOT's at Alice and Bob in LOCCNet
U_2 = torch.kron(I_4,CNOT_matrix)   #  CNOT at Bob in LOCCNet



for prob in range(0, 11, 1):   # i runs from 0 to 5

    p = torch.tensor(0 + (prob*0.05), dtype = torch.complex128)  # probability of bit-flip error of classical channels


    #########  Training for the this particular p and fixed F_in


    # This cost function is to find optimum theta
    def cost(theta):

        R_y =  R_ygate(theta)   # R_y(theta) gate
            
        U_rotation = torch.kron(torch.kron(I_2,R_y),torch.kron(I_2,R_y))

        U = torch.matmul( torch.matmul( U_rotation, U_2), U_1)  # circuit of LOCCNet

        V = permutation(U)      # Permutation A0,A1,B0,B1  to A1,B1,A0,B0

        rho_output = torch.matmul( torch.matmul(V,rho_in), torch.transpose(torch.conj(V),0,1) )



        q_00, qF_00 = q_ij_and_qF_ij(x0, rho_output)
        q_01, qF_01 = q_ij_and_qF_ij(x1, rho_output)
        q_10, qF_10 = q_ij_and_qF_ij(x2, rho_output)
        q_11, qF_11 = q_ij_and_qF_ij(x3, rho_output)



        ###########################  DECISION RULES

        # F_avg = -decisionrule11(p_training,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00/11 and rejecting 01/10

        F_avg = -decisionrule22(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00 and rejecting 01/10/11


        ############################

        # Negative sign for minimization

        return F_avg



    theta = torch.tensor(6.1, requires_grad=True)   # angle in radians

    opt = torch.optim.Adam([theta], lr = 0.01)   # lr represents learning rate

    steps = 501

    for itr in range(steps):
        opt.zero_grad()
        loss = cost(theta)
        loss.backward()
        opt.step()
        # if itr % 200 ==0:  #print for every 200 steps
        #     print('itr', itr, 'loss', loss,'theta',theta)

    theta_opt = theta

    theta_opt = theta_opt.detach()

    ############ END of training. Found the optimal theta for the particular value of p and fixed F_in







    ##############  Now use the above trained circuit for the particular value of p and fixed F_in

    R_y =  R_ygate(theta_opt)   # R_y(theta) gate
            
    U_rotation = torch.kron(torch.kron(I_2,R_y),torch.kron(I_2,R_y))

    U = torch.matmul( torch.matmul( U_rotation, U_2), U_1)  # circuit of LOCCNet

    V = permutation(U)      # Permutation A0,A1,B0,B1  to A1,B1,A0,B0

    rho_output = torch.matmul( torch.matmul(V,rho_in), torch.transpose(torch.conj(V),0,1) )


    q_00, qF_00 = q_ij_and_qF_ij(x0, rho_output)
    q_01, qF_01 = q_ij_and_qF_ij(x1, rho_output)
    q_10, qF_10 = q_ij_and_qF_ij(x2, rho_output)
    q_11, qF_11 = q_ij_and_qF_ij(x3, rho_output)



    ###########################  DECISION RULES

    # F_avg, P_success = decisionrule1(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00/11 and rejecting 01/10

    F_avg, P_success = decisionrule2(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00 and rejecting 01/10/11


    ############################

    F_opt[prob] = F_avg
    P_opt[prob] = P_success


F_opt = F_opt.detach()
P_opt = P_opt.detach()


if figure ==3:
    plt.plot(xpoints, F_opt, 'b--')

if figure ==4:
    plt.plot(xpoints, P_opt, 'b--')



################################################################################
####################### END of LOCCNet circuit, trained at various p and fixed F_in



















#################################################################
###################  Proposed NA-LOCCNet circuit, trained at various p and fixed F_in


rho = rho_Sstate(F_in)      #For S-state

# rho = rho_single_quantum_bitflip_state(F_in)  #For single quantum bit flip to ebit state


rho_in = torch.kron(rho,rho)
rho_output=torch.zeros((16,16),dtype=torch.complex128)




for prob in range(0, 11, 1):   # i runs from 0 to 5

    p = torch.tensor(0 + (prob*0.05), dtype = torch.complex128)  # probability of bit-flip error of classical channels

    #########  Training for this particular p and fixed F_in

    print('Proposed NA-LOCCNet circuit trained for F_in=0.6 at p=', p )    

        
    # This cost function is to find optimum theta for F_in and the particular p
    def cost(theta):

        U_1 = torch.kron(CNOT_matrix_reverse,CNOT_matrix_reverse)    
        
        temp_mat = two_qubit_rotation_gate(Z_gate, Y_gate, theta) #this works best
        U_rotation = torch.kron(temp_mat, temp_mat)  # this works best

        U = torch.matmul( U_rotation, U_1)  # this works best

        V = permutation(U)      #  Permutation A0,A1,B0,B1  to A1,B1,A0,B0

        rho_output = torch.matmul( torch.matmul(V,rho_in), torch.transpose(torch.conj(V),0,1) )



        q_00, qF_00 = q_ij_and_qF_ij(x0, rho_output)
        q_01, qF_01 = q_ij_and_qF_ij(x1, rho_output)
        q_10, qF_10 = q_ij_and_qF_ij(x2, rho_output)
        q_11, qF_11 = q_ij_and_qF_ij(x3, rho_output)



        ###########################  DECISION RULES

        # F_avg = -decisionrule11(p_training,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00/11 and rejecting 01/10

        F_avg = -decisionrule22(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00 and rejecting 01/10/11


        ############################

        # Negative sign for minimization

        return F_avg



    theta = torch.tensor(6.1, requires_grad=True)   # angle in radians

    opt = torch.optim.Adam([theta], lr = 0.01)   # lr represents learning rate

    steps = 501

    for itr in range(steps):
        opt.zero_grad()
        loss = cost(theta)
        loss.backward()
        opt.step()
        # if itr % 200 ==0:  #print for every 200 steps
        #     print('itr', itr, 'loss', loss,'theta',theta)

    theta_opt = theta

    theta_opt = theta_opt.detach()
    print('Trained angle for p=', p, 'is theta=', theta_opt )

    ############ END of training. Found the optimal theta for the particular value of p and fixed F_in







    ##############  Now use the above trained circuit for the particular value of p and fixed F_in

    U_1 = torch.kron(CNOT_matrix_reverse,CNOT_matrix_reverse)    
        
    temp_mat = two_qubit_rotation_gate(Z_gate, Y_gate, theta_opt)
    U_rotation = torch.kron(temp_mat, temp_mat)

    U = torch.matmul( U_rotation, U_1)

    V = permutation(U)      #  Permutation A0,A1,B0,B1  to A1,B1,A0,B0

    rho_output = torch.matmul( torch.matmul(V,rho_in), torch.transpose(torch.conj(V),0,1) )



    q_00, qF_00 = q_ij_and_qF_ij(x0, rho_output)
    q_01, qF_01 = q_ij_and_qF_ij(x1, rho_output)
    q_10, qF_10 = q_ij_and_qF_ij(x2, rho_output)
    q_11, qF_11 = q_ij_and_qF_ij(x3, rho_output)

    


    ###########################  DECISION RULES

    # F_avg, P_success = decisionrule1(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00/11 and rejecting 01/10

    F_avg, P_success = decisionrule2(p,q_00,q_01,q_10,q_11,qF_00,qF_01,qF_10,qF_11)  # Accepting 00 and rejecting 01/10/11


    ############################
    
    print('average fidelity for p=', p, 'is F_avg=', F_avg )

    F_opt[prob] = F_avg
    P_opt[prob] = P_success


F_opt = F_opt.detach()
P_opt = P_opt.detach()


if figure ==3:
    plt.plot(xpoints, F_opt, 'rd-')

if figure ==4:
    plt.plot(xpoints, P_opt, 'rd-')



################################################################################
####################### END of Proposed NA-LOCCNet circuit, trained at various p and fixed F_in








# plt.legend(["DEJMPS - F=0.6", "LOCCNet trained for p=0 and F=0.6 [paper]", " LOCCNet trained for p and F=0.6", " Proposed NA-LOCCNet trained for p and F=0.6 "], loc="lower left", fontsize=10)

# plt.xlabel("Probability of error of noisy channel", fontsize=12)



# if figure ==3:
#     plt.ylabel("Average output fidelity", fontsize=12)

# if figure ==4:
#     plt.ylabel("Probability of success", fontsize=12)

plt.grid(True)
plt.tight_layout()

# plt.title("Proposed circuit trained at p = 0.5 and F=0.6")

plt.show()

