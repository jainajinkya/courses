function [m_new,s,sig_new] = belief_dyna(A,B,C,W,m,s,sig,u)
m_new = A*m + B*u + normrnd(0,s,[2,1]);
    
% Covariance Dynamics
gamma = A*sig*A';
dsig = sig*(A+A') - gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A'))) + ...
    gamma*C'*((C*gamma*C' + W)\(C*sig*(A+A')*C'))*((C*gamma*C' + W)\(C*gamma))...
    - sig*(A+A')*C'*((C*gamma*C' + W)\(C*gamma));

s = dsig(1,1);
sig_new = gamma - gamma*C'*((C*gamma*C' + W)\(C*gamma));
end