%MAIN
%
% Demonstrates how to solve the finite-horizon continuous-time linear
% quadratic regulator problem for a linear system
%
clear variables
close all
clc

nState = 2;
nInput = 2;
nSoln = 100;
tol = 1e-6;

%LTI plant
s = 0.2*ones(3,1);
x = [0; 2.5];
m = [2.5 ;2.5];
t_f = 100;

traj = [x];
cov = [s(1)];

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];
Q = 0.5*eye(nState);
R = 0.5*eye(nInput);
F = 200*eye(nState);  %Terminal Cost


sig = s(1)*eye(nState);

A_ext = [[A'; zeros(1,nState)]';0 0 0]';
B_ext = [B; 0 0];
Q_ext = [[Q'; zeros(1,nState)]';0 0 0]' ; 
R_ext = [R; 0 0];

Q_large = 20;
Lambda = 200;
S = zeros(3,3,t_f);
S(:,:,t_f)= [Q_large 0 0; 0 Q_large 0; 0 0 Lambda];

for t=t_f-1:-1:1
   % Control law
    A_ext(3,3) = 1;
    S(:,:,t)= Q_ext + A_ext'*S(:,:,t+1)*A_ext - A_ext'*S(:,:,t+1)*B_ext*((B_ext'*S(:,:,t+1)*B_ext+ R)\(B_ext'*S(:,:,t+1)*A_ext)); 
    
end

for t= 1:t_f
     u = -(B_ext'*S(:,:,t)*B_ext + R)\(B_ext'*S(:,:,t)*A_ext)*[m;0.1];
    
    % Dynamics
    x = A*(x-m) + (m + u);
    m = m + u ;
    % Next Interation
    W = normpdf(x,0,(0.5*(5 - x(1))^2 + 5));
    W = W(1)*eye(2);
    gamma = A*sig*A';
    sig = gamma - gamma*C'*(C*gamma*C' + W)\C*gamma 
    
    traj = [traj,x];
    cov =  [cov;sig(1,1)];
end

figure(1);clf;
scatter(traj(1,:),traj(2,:),'r');

figure(2); clf;
scatter(1:size(traj,2),cov);


%Make a plot showing what the gains look like:
% figure(21); clf;
% KK = reshape([Soln.K],nState,nSoln);
% t = [Soln.t];
% for i=1:nState
%     subplot(nState,1,i); hold on;
%    plot(tSpan,[K(i), K(i)],'k--','LineWidth',2)
%    plot(t,KK(i,:),'r-','LineWidth',2);
%    ylabel(['K(' num2str(i) ')']);
% end
% xlabel('Time')
%
% subplot(nState,1,1);
% title('Finite-Horizon Gains, Compare to Infinite Horizon Soln')