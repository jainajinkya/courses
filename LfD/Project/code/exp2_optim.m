% Direct Transcription
clear
close
clc

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
u = [0,0]';

m_con = rand(nState,1,k);
s_con = rand(1,1,k);
sig_con = rand(nState,nState,k);

m_dash = rand(nState,1,k);
u_dash = rand(nInput,1,k);
sig_dash = rand(nState,nState,k);
s_dash = rand(k);

A = [1 0; 0 1];
B = [1 0; 0 1];
C = [1 0; 0 1];
Q = 0.1*eye(nState);
R = 1.0*eye(nInput);
Q_f = 500*eye(nState);
labda = 2000;
W = 0.5*(5.0-m(1))^2*eye(nOutput);


for j=1:10

    x0 = [reshape(m_dash(:,:,1:end),[(k)*nState,1]);reshape(sig_dash(1,1,:),[k,1]);reshape(u_dash,[k*nInput,1])];
    
    %Constraints
    for i=1:k
        m = m_con(:,:,i);
        s = s_con(i);
        sig = sig_dash(:,:,i);
        u = u_dash(:,:,i);
        [m,s,sig] = phi(A,B,C,W,m,s,sig,u,del);
        m_con(:,:,k) = m;
        sig_con(:,:,k) = sig;
    end
    m_con(:,:,k) = goal;
    
    % Defining fmincon
    % % % % % optimization vector := [m1,.., mk [nState each], s, u1,..,uk [nInput each]]
    opti_A = [];
    opti_B = [];
    opti_Aeq = [[eye(k*(nState+1)), zeros(k*(nState+1),nInput*k)]; zeros(k*nInput,k*(nState+1+nInput))];
    opti_Beq = [m;reshape(m_con(:,:,1:end-1),[(k-1)*nState,1]);reshape(sig_con(1,1,:),[k,1]);zeros(k*nInput,1)];
    x = fmincon(@(x)obj_fn(x),x0,opti_A,opti_B,opti_Aeq,opti_Beq);
    
    s_dash = x(2*k+1:3*k);
    for i=1:2:2*k
        m_dash(:,ceil(i/2)) = [x(i),x(i+1)]';
        u_dash(:,ceil(i/2)) = [x(2*k+k+i),x(2*k+k+i+1)]';
        sig_dash(:,:,ceil(i/2)) = s_dash(ceil(i/2))*eye(2,2);
    end
end



