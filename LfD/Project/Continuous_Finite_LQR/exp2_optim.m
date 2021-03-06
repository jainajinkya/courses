% Direct Transcription
nState = 2;
nInput = 2;
nOutput = 2;
nCov = nState^2;

T = 30;
del = 5;
k = round(T/del);

m = [2,2]';
sig = 5*eye(nState);
s = sig(1,1);
goal = [0,0]';

% These values neeed not to defined explicitely: Just for reference
m_dash = zeros(nState,1,k);
s_dash = zeros(1,1,k);
sig_dash = zeros(nState,1,k);
% u_dash = zeros(nInput,1,k-1);

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];
Q = 0.1*eye(nState);
R = 1.0*eye(nInput);
Q_f = 500*eye(nState);
labda = 2000;

%Constraints
for i=1:k
   [m_dash(:,:,k),s_dash(:,:,k),sig_dash(:,:,k)] = phi(A,B,C,W,m,sig,u,delta);
   m = m_dash(:,:,k);
   sig = sig_dash(:,:,k);   
end
m(:,:,k) = goal;

% Cost
for i=1:k
   J = m_dash(:,:,k)'*Q*m_dash(:,:,k) + u_dash(:,:,k)'*R*u_dash(:,:,k);
end

% Defining fmincon
x = [[]];

J = J + s_F'*labda*s_F;
