function J = obj_fn(x)
k = 6;
Q = 0.1*eye(2);
R = 1.0*eye(2);
labda = 2000;

s = x(2*k+k);
for i=1:2:k
    m = [x(i),x(i+1)]';
    u = [x(2*k+k+i),x(2*k+k+i+1)]';
    J = m'*Q*m + u'*R*u;
end
J = J + s'*labda*s;

end