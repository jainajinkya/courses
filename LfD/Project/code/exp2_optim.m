% Direct Transcription
clear
close
clc

nState = 2;
nInput = 2;
nOutput = 2;
nCov = nState^2;

T = 50;
del = 10;
k = round(T/del);

m_dash = rand(nState,1,k);
u_dash = rand(nInput,1,k);
sig_dash = rand(nState,nState,k);
s_dash = rand(k,1);

m = [2,2]';
sig = 5*eye(nState);
s = sig(1,1);
goal = [0,0]';
u = [0,0]';

m_dash(:,:,1) = m;
sig_dash(:,:,1) = sig;
s_dash(1,1) = s;
u_dash(:,:,1) = u;

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];
Q = 0.1*eye(nState);
R = 1.0*eye(nInput);
Q_f = 500*eye(nState);
labda = 2000;
W = 0.5*(5.0-m(1))^2*eye(nOutput);

%Constraints
for i=1:k-1
    [m_dash(:,:,i+1),s_dash(i+1,1),sig_dash(:,:,i+1)] = phi(A,B,C,W,m_dash(:,:,i),s_dash(i,1),sig_dash(:,:,i),u_dash(:,:,i),del);
end
m_dash(:,:,end) = goal;

% x0 = [reshape(m_dash(:,:,1:end),[(k)*nState,1]);reshape(sig_dash(1,1,:),[k,1]);reshape(u_dash,[k*nInput,1])];
x0 = rand(k*(nState + nInput + 1),1);


% Defining fmincon
% % % % % optimization vector := [m1,.., mk [nState each], s, u1,..,uk [nInput each]]
opti_A = [];
opti_B = [];
% opti_Aeq = [[eye(k*(nState+1)), zeros(k*(nState+1),nInput*k)]; zeros(k*nInput,k*(nState+1+nInput))];
opti_Aeq = [zeros(nState,k*(nState+1+nInput));zeros((k-1)*nState,nState),eye((k-1)*nState),zeros((k-1)*nState,(k+1)*nInput);...
    zeros(k,k*nState+1),[zeros(1,k-1);eye(k-1)],zeros(k,k*nInput);zeros(nInput,k*(nState+1+nInput))];
opti_Beq = [m;reshape(m_dash(:,:,1:end-1),[(k-1)*nState,1]);reshape(sig_dash(1,1,:),[k,1]);zeros(k*nInput,1)];
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
[xfinal fval exitflag] = fmincon(@(x)obj_fn(x,k,Q,R,labda),x0,opti_A,opti_B,opti_Aeq,opti_Beq,options)

%     s_dash = x(2*k+1:3*k);
%     for i=1:2:2*k
%         m_dash(:,ceil(i/2)) = [x(i),x(i+1)]';
%         u_dash(:,ceil(i/2)) = [x(2*k+k+i),x(2*k+k+i+1)]';
%         sig_dash(:,:,ceil(i/2)) = s_dash(ceil(i/2))*eye(2,2);
%     end
% end



