function xNew = simulator(x,mu,u)
global nState mA mB chpts

idx = size(chpts,2)*ones(nState,1);

for i = 1:nState
    for j = size(chpts,2):-1:1
        if(x(i,1) <= chpts(i,j))
            idx(i,1) = j  ;      
        end
    end
end

mode = max(idx);
A = mA(:,:,mode);
B = mB(:,:,mode);

xNew = A*(x-mu) + (A*mu + B*u);

end
