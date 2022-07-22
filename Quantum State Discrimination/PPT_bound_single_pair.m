% PPT Bound on success probability in Quantum state discrimination (QSD) problem 
% distinguishing two states(|phi+> and state obtained by passing |phi-> through ampltidue damping channel)
% given one pair of qubits
% using MATLAB cvx package

clc
clear all
close all


rho_1 = [0.5 0 0 0.5; 0 0 0 0; 0 0 0 0; 0.5 0 0 0.5];
n=size(rho_1,1);

% PPT Bound

prob_succ_PPT = zeros(1,11);
for i = 1:1:11
    g = (i-1)/10;
    % output state by passing |phi->'s two qubits through amplitude damping
    % channel
    rho_2 = [(1+(g^2))/2 0 0 (g-1)/2; 0 (g-(g^2))/2 0 0; 0 0 (g-(g^2))/2 0; (g-1)/2 0 0 ((1-g)^2)/2];

    cvx_begin sdp
         variable M_1(n,n) hermitian semidefinite;
         variable M_2(n,n) hermitian semidefinite;
         maximize ( 0.5*(trace(M_1*rho_1 + M_2*rho_2)) )
         subject to
             M_1 + M_2 == eye(4);
             partial_transpose(M_1,2,2) == semidefinite(n);
             partial_transpose(M_2,2,2) == semidefinite(n);

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
    rho_2 = [(1+(g^2))/2 0 0 (g-1)/2; 0 (g-(g^2))/2 0 0; 0 0 (g-(g^2))/2 0; (g-1)/2 0 0 ((1-g)^2)/2];

    cvx_begin sdp
         variable M_1(n,n) hermitian semidefinite;
         variable M_2(n,n) hermitian semidefinite;
         maximize ( 0.5*(trace(M_1*rho_1 + M_2*rho_2)) )
         subject to
             M_1 + M_2 == eye(4);
%              partial_transpose(M_1) == semidefinite(n);
%              partial_transpose(M_2) == semidefinite(n);

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
ylim([0.85,1])
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















% 
% % This function is manual
% function N = partial_transpose(M)
%   
%     % cells that wont change
%     N(1,1) = M(1,1);
%     N(2,2) = M(2,2);
%     N(3,3) = M(3,3);
%     N(4,4) = M(4,4);
%     N(1,3) = M(1,3);
%     N(2,4) = M(2,4);
%     N(3,1) = M(3,1);
%     N(4,2) = M(4,2);
%     
%     % Cells that change
%     N(1,2) = M(2,1);
%     N(2,1) = M(1,2);
% 
%     N(1,4) = M(2,3);
%     N(2,3) = M(1,4);
% 
%     N(3,2) = M(4,1);
%     N(4,1) = M(3,2);
% 
%     N(3,4) = M(4,3);
%     N(4,3) = M(3,4);
% 
% end









% cvx_begin sdp
%     variable M_1(4,4) complex 
%     variable M_2(4,4) complex 
%     dual variable Q
%     minimize( norm( Z - P, 'fro' ) )
%     Z >= 0 : Q;
% cvx_end