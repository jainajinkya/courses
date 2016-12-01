function J = obj_fn(x,nState,nSegments,Q,R,labda,goal)
global nInput
s = x((nState+1)*nSegments,1);
for i=1:nState:nSegments*nState
    m = [x(i:i+nState-1,1)];
    u = [x(nState*nSegments+nSegments+i: nState*nSegments+nSegments+i+ nInput-1,1)];
    J = (m-goal)'*Q*(m-goal) + u'*R*u;
%      J = m'*Q*m + u'*R*u;
end
J = J + s'*labda*s;

end