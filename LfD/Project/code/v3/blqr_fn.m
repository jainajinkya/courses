function u = blqr_fn(m,sig,u_bar,m_bar,sig_bar,A,B,C,T)
% As the optimal action is to be calculated just for the most probable
% model, hence, there is no need to pass the system matrices for all the
% models. 
global nState nInput Q R labda

Q_f = 20*eye(nState);
Qnew = Q;
Rnew = R;


% Extended Matrices
A_ext = [[A'; zeros(1,nState)]';zeros(1,nState+1)]';
B_ext = [B;zeros(1,nInput)];
Q_ext = [[Qnew'; zeros(1,nState)]';zeros(1,nState+1)]' ;
F = [[Q_f'; zeros(1,nState)]';zeros(1,nState+1)]';  %Terminal Cost
F(nState+1,nState+1) = labda;

%% LQR Control
% W = 0.5*(5.0-x(1))^2*eye(nOutput);
W = 0.5*eye(nState);
gamma = A*sig*A';
dg_dsig= A*A';
ds_dsig = dg_dsig - dg_dsig*C'*(inv(C*gamma*C' + W)*(C*gamma)) ...
    + gamma*C'*(inv(C*gamma*C' + W)*(C*dg_dsig*C')*inv(C*gamma*C' + W)*(C*gamma)) ...
    - gamma*C'*(inv(C*gamma*C' + W)*(C*dg_dsig));

% d_s_m = gamma*C'*(inv(C*gamma*C' + W)*((m(1)-5)*eye(size(W)))*inv(C*gamma*C' + W)*(C*gamma));
d_s_m = 0;

A_ext(nState+1,:) = [d_s_m(1,1) zeros(1,nState-1) ds_dsig(1,1)];

[K,S3] = finiteLQR(T,A_ext,B_ext,Q_ext,Rnew,F);

u = -K(:,:,1)*[(m-m_bar);(sig(1,1)-sig_bar)] + u_bar;
end