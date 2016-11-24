function J = obj_fn(x,nState,nSegments,Q,R,labda,goal)
s = x((nState+1)*nSegments,1);
for i=1:2:nSegments*nState
    m = [x(i),x(i+1)]';
    u = [x(2*nSegments+nSegments+i),x(2*nSegments+nSegments+i+1)]';
%     J = (m-goal)'*Q*(m-goal) + u'*R*u;
     J = m'*Q*m + u'*R*u;
end
J = J + s'*labda*s;

end