function J = obj_fn(x,k,Q,R,labda)
s = x(2*k+k);
for i=1:2:k
    m = [x(i),x(i+1)]';
    u = [x(2*k+k+i),x(2*k+k+i+1)]';
    J = m'*Q*m + u'*R*u;
end
J = J + s'*labda*s;

end