function u = blqr_fn(nState,nOutput,t_f,x,m,sig,A,B,C,Q,R,labda)
% As the optimal action is to be calculated just for the most probable
% model, hence, there is no need to pass the system matrices for all the
% models. 

t_res = 1;

Q_f = 20*eye(nState);

% Extended Matrices
A_ext = [[A'; zeros(1,nState)]';0 0 0]';
B_ext = [B; 0 0];
Q_ext = [[Q'; zeros(1,nState)]';0 0 10]' ;
F = [[Q_f'; zeros(1,nState)]';0 0 labda]';  %Terminal Cost

%% LQR Control
W = 0.5*(5.0-x(1))^2*eye(nOutput);
gamma = A*sig*A';
dg_dsig= A*A';
ds_dsig = dg_dsig - dg_dsig*C'*(inv(C*gamma*C' + W)*(C*gamma)) ...
    + gamma*C'*(inv(C*gamma*C' + W)*(C*dg_dsig*C')*inv(C*gamma*C' + W)*(C*gamma)) ...
    - gamma*C'*(inv(C*gamma*C' + W)*(C*dg_dsig));

d_s_m = gamma*C'*(inv(C*gamma*C' + W)*((m(1)-5)*eye(size(W)))*inv(C*gamma*C' + W)*(C*gamma));

A_ext(3,:) = [d_s_m(1,1) 0  ds_dsig(1,1)];

[K,S3] = finiteLQR(t_f,A_ext,B_ext,Q_ext,R,F,t_res);

u = -K(:,:,1)*[m;sig(1,1)];
end