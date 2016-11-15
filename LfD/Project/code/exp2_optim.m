% Direct Transcription
% clear
close
clc

nState = 2;
nInput = 2;
nOutput = 2;
nCov = nState^2;

T = 5;
del = 1;
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
Q = 0.5*eye(nState);
R = 0.5*eye(nInput);
labda = 200;
W = 0.5*(5.0-m(1))^2*eye(nOutput);

%Constraints
for i=1:k-1
    [m_dash(:,:,i+1),s_dash(i+1,1),sig_dash(:,:,i+1)] = phi(A,B,C,W,m_dash(:,:,i),s_dash(i,1),sig_dash(:,:,i),u_dash(:,:,i),del);
end
m_dash(:,:,end) = goal;

% x0 = [reshape(m_dash(:,:,1:end),[(k)*nState,1]);reshape(sig_dash(1,1,:),[k,1]);reshape(u_dash,[k*nInput,1])];
x0 = zeros(k*(nState + nInput + 1),1) - 0.25;
% x0 = xfinal;


% Defining fmincon
% % % % % optimization vector := [m1,.., mk [nState each], s, u1,..,uk [nInput each]]
opti_A = [];
opti_B = [];
opti_Aeq = [eye(nState), zeros(nState,(k-1)*nState+k*(1+nInput));zeros((k-1)*nState,nState),eye((k-1)*nState),zeros((k-1)*nState,k*(nInput+1));...
    zeros(k,k*nState),eye(k),zeros(k,k*nInput);zeros(k*nInput,k*(nState+1+nInput))];
opti_Beq = [reshape(m_dash(:,:,1:end),[k*nState,1]);reshape(s_dash(:,1),[k,1]);zeros(k*nInput,1)];

lb = -50*ones(size(x0,1),1);
ub = 50*ones(size(x0,1),1);
options = optimoptions('fmincon','Display','iter','Algorithm','sqp','ConstraintTolerance',1e-12);
% [xfinal,fval,exitflag] = fmincon(@(x)obj_fn(x,k,Q,R,labda),x0,opti_A,opti_B,opti_Aeq,opti_Beq,lb,ub,[],options);
[xfinal,fval,exitflag] = fmincon(@(x)obj_fn(x,k,Q,R,labda),x0,opti_A,opti_B,opti_Aeq,opti_Beq,lb,ub,[]);




%%%%%%%%%%%Visualization%%%%%%%%%
traj = [];
for i=1:2:k*nState
   traj = [traj, [xfinal(i,1);xfinal(i+1,1)]]; 
end

figure(1);clf;
hold on
plot(traj(1,1:end-1),traj(2,1:end-1),'r--','LineWidth',2);
% scatter(traj(1,:),traj(2,:));
plot(traj(1,1),traj(2,1),'mo', 'markers',12);
plot(traj(1,end-1),traj(2,end-1),'gx','markers',12);
hold off

traj

%     s_dash = x(2*k+1:3*k);
%     for i=1:2:2*k
%         m_dash(:,ceil(i/2)) = [x(i),x(i+1)]';
%         u_dash(:,ceil(i/2)) = [x(2*k+k+i),x(2*k+k+i+1)]';
%         sig_dash(:,:,ceil(i/2)) = s_dash(ceil(i/2))*eye(2,2);
%     end
% end




