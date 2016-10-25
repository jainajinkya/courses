%MAIN 
%
% Demonstrates how to solve the finite-horizon continuous-time linear
% quadratic regulator problem for a linear system
%

nState = 2;
nInput = 2;
nSoln = 100;

%LTI plant
% A = 0.5*eye(nState) + 0.5*rand(nState);
% B = rand(nState,nInput);
A = zeros(nState + nState^2);
B = zeros(nInput + nState^2, nInput);

A(1,1) = 1;
A(2,2) = 1;

%LTI cost
Q = 0.5*eye(nState);
R = 0.5*eye(nInput);
F = 200*eye(nState);  %Terminal Cost

%Solve continuous LQR problem:
% [K,S,E] = lqr(A,B,Q,R);

%Solve the finite-horizon version
tSpan = [0,30];  %Large time...
tol = 1e-6;
Soln = finiteLqr(tSpan,A,B,Q,R,F,nSoln,tol);

%Compare the results:  (Should go to zero for tSpan(2) goes to infinity)
K_error = K - Soln(1).K
S_error = S - Soln(1).S
E_error = E - Soln(1).E

%Make a plot showing what the gains look like:
figure(21); clf;
KK = reshape([Soln.K],nState,nSoln);
t = [Soln.t];
for i=1:nState
    subplot(nState,1,i); hold on;
   plot(tSpan,[K(i), K(i)],'k--','LineWidth',2) 
   plot(t,KK(i,:),'r-','LineWidth',2);
   ylabel(['K(' num2str(i) ')']);
end
xlabel('Time')

subplot(nState,1,1); 
title('Finite-Horizon Gains, Compare to Infinite Horizon Soln')