% PPT Bound on success probability in Quantum state discrimination (QSD) problem 
% distinguishing two states(|phi+> and state obtained by passing |phi-> through ampltidue damping channel)
% given two pair of qubits
% using MATLAB cvx package

clc
clear all
close all


phi_plus = [0.5 0 0 0.5; 0 0 0 0; 0 0 0 0; 0.5 0 0 0.5];
rho_1=permutation(kron(phi_plus,phi_plus));

n=size(rho_1,1);

% PPT Bound

prob_succ_PPT = zeros(1,11);
for i = 1:1:11
    g = (i-1)/10;
    % output state by passing |phi->'s two qubits through amplitude damping
    % channel
    A_phi_minus = [(1+(g^2))/2 0 0 (g-1)/2; 0 (g-(g^2))/2 0 0; 0 0 (g-(g^2))/2 0; (g-1)/2 0 0 ((1-g)^2)/2];
    rho_2=permutation(kron(A_phi_minus,A_phi_minus));

    cvx_begin sdp
         variable M_1(n,n) hermitian semidefinite;
         variable M_2(n,n) hermitian semidefinite;
         maximize ( 0.5*(trace(M_1*rho_1 + M_2*rho_2)) )
         subject to
             M_1 + M_2 == eye(16);
             partial_transpose(M_1,4,4) == semidefinite(n);
             partial_transpose(M_2,4,4) == semidefinite(n);

    cvx_end
    prob_succ_PPT(i) = cvx_optval;
end

xaxis_coordinates = 0:0.1:1;
figure
% plot( xaxis_coordinates, prob_succ_PPT,'--o', 'Linewidth',2,'Color', [1, 0, 0] )
plot( xaxis_coordinates, prob_succ_PPT, 'Linewidth',2, 'DisplayName','PPT')
hold on


% POVM Bound

prob_succ_POVM = zeros(1,11);
for i = 1:1:11
    g = (i-1)/10;
    A_phi_minus = [(1+(g^2))/2 0 0 (g-1)/2; 0 (g-(g^2))/2 0 0; 0 0 (g-(g^2))/2 0; (g-1)/2 0 0 ((1-g)^2)/2];
    rho_2=permutation(kron(A_phi_minus,A_phi_minus));

    cvx_begin sdp
         variable M_1(n,n) hermitian semidefinite;
         variable M_2(n,n) hermitian semidefinite;
         maximize ( 0.5*(trace(M_1*rho_1 + M_2*rho_2)) )
         subject to
             M_1 + M_2 == eye(16);
%              partial_transpose(M_1,4,4) == semidefinite(n);
%              partial_transpose(M_2,4,4) == semidefinite(n);

    cvx_end
    prob_succ_POVM(i) = cvx_optval;
end


% plot( xaxis_coordinates, prob_succ_POVM,'-', 'Linewidth',2,'Color', [0, 0, 1] )
plot( xaxis_coordinates, prob_succ_POVM, 'Linewidth',2, 'DisplayName','POVM')
% hold off







legend


xlabel('Noise Parameter ','fontsize',20)
ylabel('Average probability of success','fontsize',20)
set(gca,'FontName','Times New Roman','FontSize',20);
ylim([0.932,1])
axis square





% This function is automatic. works for any matrix orders. 
% a and b represents dimensions of Alice and Bob
function N = partial_transpose(M,a,b)
    for R = 1:a
        for C = 1:a
            for i = 1:b
                for j = 1:b
                    temp(i,j) = M(b*(R-1)+i,b*(C-1)+j);
                end
            end
            temp = temp';
            for i = 1:b
                for j = 1:b
                    N(b*(R-1)+i,b*(C-1)+j) = temp(i,j);
                end
            end
            clear temp
        end
    end
end



% permutation funtion A1,B1,A0,B0 to A0,A1,B0,B1    (for 16 X 16 matrix)
function matrix2= permutation(matrix1)
    prm = [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16];
    matrix2 = zeros(16,16);
    for i = 1: 16  % i runs from 1 to 16 with increments of 1
        for j = 1: 16
            matrix2(prm(i),prm(j)) = matrix1(i,j);
        end
    end
end
    
    
