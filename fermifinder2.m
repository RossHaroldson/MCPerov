function [n, p, Naion, Ndion, Ef] = fermifinder2(m_e, m_h, Na, Eta, qa, Nd, qd, Etd, Egap, T)
% Uses the bisection method to find the Fermi level satisfying charge neutrality.
m_0   = 9.109534e-31; kb = 1.380662e-23;
hbar = 1.0545887e-34; q = 1.6021892e-19; S = 1;
h = 2*pi*hbar;
F = @(x,c) x.^(1/2)./(1+exp(x-c));
itmax = 10000; tol = Egap/1e18;
a = 0; b = Egap; c = (a+b)/2;
% Compute electron and hole densities at lower bound
na = 2*(2*pi*m_e*m_0*kb*T/(h^2))^(3/2)*integral(@(x)F(x,(a-Egap)/(kb*T)),0,inf);
pa = 2*(2*pi*m_h*m_0*kb*T/(h^2))^(3/2)*integral(@(x)F(x,-a/(kb*T)),0,inf);
Ndiona = qd.*Nd .* (1 - 1./(1+S*exp((Etd - a)/(kb*T))));
Naiona = qa.*Na .* (1 - 1./(1+S*exp((Eta - (Egap-a))/(kb*T))));
diffa = pa - na - sum(Naiona) + sum(Ndiona);
itr = 1; 
while itr < itmax
    c = (a+b)/2;
    nc = 2*(2*pi*m_e*m_0*kb*T/(h^2))^(3/2)*integral(@(x)F(x,(c-Egap)/(kb*T)),0,inf);
    pc = 2*(2*pi*m_h*m_0*kb*T/(h^2))^(3/2)*integral(@(x)F(x,-c/(kb*T)),0,inf);
    Ndionc = qd.*Nd .* (1 - 1./(1+S*exp((Etd - c)/(kb*T))));
    Naionc = qa.*Na ./ (1+S*exp((Eta - c)/(kb*T)));
    diffc = pc - nc - sum(Naionc) + sum(Ndionc);
    if diffc == 0 || (b-a)/2 < tol
        Ef = c; n = nc; p = pc; Naion = Naionc; Ndion = Ndionc;
        disp('Reached convergence');
        return
    end
    itr = itr + 1;
    if sign(diffc) == sign(diffa)
        a = c;
        diffa = diffc;
    else
        b = c;
    end
end
disp('Reached iteration limit');
Ef = c; n = nc; p = pc; Naion = Naionc; Ndion = Ndionc;
end
