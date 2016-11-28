function [u_plan,mu_plan,s_plan] = createPlan(mu,cov,nState,nInput,nGauss,nSegments)

x0 = rand(((nState+1)*nGauss +nInput)*nSegments,1);
x0(1:nState*nGauss,1) = reshape(mu,[nState*nGauss,1]);
x0(nState*nGauss*nSegments+1:nState*nGauss*nSegments+nGauss,1) = cov(1,1,:);
x0((nState+1)*nGauss*nSegments+1:(nState+1)*nGauss*nSegments+nInput,1) = u0;

opti_A = [];
opti_B = [];
opti_Aeq = [];
opti_Beq = [];
lb = -20*ones(size(x0,1),1);
ub = 20*ones(size(x0,1),1);
constraintRelax = 10.0;
exitflag = 0;
fnEval = 100000;

while(exitflag~=1)
    nonlcon = @(x)covCons2(x,x0,mA,mB,mC,wts,nGauss,nInput,nSegments,delta,constraintRelax);
    
    options = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxFunctionEvaluations',fnEval);
    % options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
    [xfinal,fval,exitflag] = fmincon(@(x)obj_fn(x,nState,nSegments,Q,R,labda,goal),x0,opti_A,opti_B,opti_Aeq,opti_Beq,lb,ub,nonlcon,options);
    
    if(exitflag==0)
        x0 = xfinal;
        constraintRelax = constraintRelax/2;
        fnEval = 2*fnEval;
    end
end

mu_plan = xfinal(1:nState*nGauss*nSegments,:);
s_plan = xfinal(nState*nGauss*nSegments+1:(nState+1)*nGauss*nSegments,:);
u_plan = xfinal((nState+1)*nGauss*nSegments+1:end);

mu_plan = reshape(mu_plan,[nState,nGauss,nSegments]);
s_plan = reshape(s_plan,[1,nGauss,nSegments]);
u_plan = reshape(u_plan,[nInput,nSegments]);

end