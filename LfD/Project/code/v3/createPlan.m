function [u_plan,mu_plan,s_plan] = createPlan(mu,cov)
global nState nInput nModel nGauss T goal Q R labda mA mB mC

nSegments = 3;
delta = round(T/nSegments);

wts = (1/nModel)*ones(nModel,1);

x0 = zeros(((nState+1)*nGauss +nInput)*nSegments,1);
x0(1:nState*nGauss,1) = reshape(mu,[nState*nGauss,1]);
x0(nState*nGauss*nSegments+1:nState*nGauss*nSegments+nGauss,1) = cov(1,1,:);
% x0((nState+1)*nGauss*nSegments+1:(nState+1)*nGauss*nSegments+nInput,1) = u0;

opti_A = [];
opti_B = [];
opti_Aeq = [];
opti_Beq = [];
lb = -20*ones(size(x0,1),1);
ub = 20*ones(size(x0,1),1);
constraintRelax = 0.0;
exitflag = 0;
fnEval = 100000;

while(1)
    nonlcon = @(x)covCons2(x,x0,mA,mB,mC,wts,nGauss,nInput,nSegments,delta,constraintRelax);
    
%     options = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunctionEvaluations',fnEval);
    options = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunEvals',fnEval);
    [xfinal,fval,exitflag] = fmincon(@(x)obj_fn(x,nState,nSegments,Q,R,labda,goal),x0,opti_A,opti_B,opti_Aeq,opti_Beq,lb,ub,nonlcon,options);
   
    exitflag
    
    if(exitflag==0)
        x0 = xfinal;
        constraintRelax = constraintRelax/2;
        fnEval = 2*fnEval;
    elseif(exitflag==-2)
        constraintRelax = 5;
        fnEval = 2*fnEval;
        clear xfinal
    elseif(exitflag==2 && constraintRelax ~= 0.0)
        x0 = xfinal; % + rand()*ones(size(x0,1),1);
        constraintRelax = constraintRelax/2;
    else
        break;
    end
end

mu_new = xfinal(1:nState*nGauss*nSegments,:);
s_new = xfinal(nState*nGauss*nSegments+1:(nState+1)*nGauss*nSegments,:);
u_new = xfinal((nState+1)*nGauss*nSegments+1:end);

mu_new = reshape(mu_new,[nState,nGauss,nSegments]);
s_new = reshape(s_new,[1,nGauss,nSegments]);
u_new = reshape(u_new,[nInput,nSegments]);

mu_plan = zeros(nState,nGauss,T+1);
s_plan = zeros(1,nGauss,T+1);
u_plan = zeros(nInput,T);

mu_plan(:,:,1) = mu_new(:,:,1);
s_plan(:,:,1) = s_new(:,:,1);

for i=1:nGauss
    cov(:,:,i) = s_plan(:,i,1)*eye(nState);
end
x = [0,0]';
for i=1:nSegments    
    for t=1:delta
        [mu_plan(:,:,(i-1)*delta+t+1),cov] = modelDynamics(mu_plan(:,:,(i-1)*delta+t),mu_plan(:,:,(i-1)*delta+t),cov,u_new(:,i),mA,mB,mC,wts);
        [wts] = model_wts(mu_plan(:,:,(i-1)*delta+t+1),cov);
        s_plan(:,:,(i-1)*delta+t+1) = cov(1,1);
        u_plan(:,(i-1)*delta+t+1) = u_new(:,i);
    end
end

end