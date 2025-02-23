function [qFn, qFp] = quasifermifind(e_density,h_density,m_e,m_h,Egap)
m_0   = 9.109534e-31;    % free electron mass (kg)
kb   = 1.380662e-23;
hbar = 1.0545887e-34;
T=300;
h = hbar*2*pi;
F= @(x,c) x.^(1/2)./(1+exp(x-c));
itmax = 5000;
tol = Egap/1e17;
itr=1;
ea=0; eb = Egap;
% for electrons
na=2*(2*pi*m_e*m_0*kb*T/h^2)^(3/2).*integral(@(x)F(x,(ea-Egap)/kb/T),0,inf);
ediffa = e_density-na;
while itr < itmax
    ec=(ea+eb)/2;
    % for electrons
    nc=2*(2*pi*m_e*m_0*kb*T/h^2)^(3/2).*integral(@(x)F(x,(ec-Egap)/kb/T),0,inf);
    ediffc = e_density-nc;
    if ediffc == 0 || (eb-ea)/2 < tol
        qFn=ec;
        break
    end
    itr=itr+1;
    if sign(ediffc) == sign(ediffa)
        ea=ec;
        ediffa=ediffc;
    else
        eb=ec;
    end
end
% for holes
itr=1;
ha=0; hb = Egap;
pa=2*(2*pi*m_h*m_0*kb*T/h^2)^(3/2).*integral(@(x)F(x,(0-ha)/kb/T),0,inf);
hdiffa = h_density-pa;
while itr < itmax
    hc=(ha+hb)/2;
    % for holes
    pc=2*(2*pi*m_h*m_0*kb*T/h^2)^(3/2).*integral(@(x)F(x,(0-hc)/kb/T),0,inf);
    hdiffc = h_density-pc;
    if hdiffc == 0 || (hb-ha)/2 < tol
        qFp=hc;
        break
    end
    itr=itr+1;
    if sign(hdiffc) == sign(hdiffa)
        ha=hc;
        hdiffa=hdiffc;
    else
        hb=hc;
    end
end
qFn=ec;qFp=hc;

end