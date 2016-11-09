function [m_new,s,sig_new] = phi(A,B,C,W,m,s,sig,u,delta)

m_init = m;
s_init = sig(1,1);
sig_init = sig;

for i=1:delta
    [m_new,s,sig_new] = belief_dyna(A,B,C,W,m,s,sig,u);
    m = m_new;
    sig = sig_new;
end

m_new = m_new + m_init;
s = s + s_init;
sig_new = sig_new + sig_init;
end
