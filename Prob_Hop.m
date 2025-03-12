function P = Prob_Hop(E_DB,F,aa,E_th,kvector,m,Z,q)
% This function returns the probability of a defect hoping based off the
% Diffusion barrier energy (E_DB), Electric potential shift based off the
% electric field strength vector F=[Fx Fy Fz] from applied bias, Lattice
% vectors to available nearest neighbor sites a=[ ax1 ay1 az1; ax2 ay2 az2;
% ...; ax6 ay6 az6], Thermal energy E_th = k_B*T, Momentum vector of the
% carrier kvector, the Ionization state of the defect Z (...,+2,+1,0,-1,-2,...),%
% and the type of electronic carrier q (+1 for holes and -1 for electrons).
% All units are in SI.
% NOTES:
% I included the type of carrier (hole or electron) because if there are
% two like charges you need to repel.

% Physical constants
eps_0 = 8.854187818e-12;
m_0   = 9.109534e-31;
hbar  = 1.0545887e-34;
h     = hbar*2*pi;
kb    = 1.380662e-23;
a0    = 5.2917706e-11;
e     = 1.6021892e-19;

if m == 0
    m = 1;
end
if Z ~= 0
    Zsign = Z / abs(Z);
else
    Zsign = Z;
end

% Boltzmann factor for a given lattice vector
B_fac = @(E_DB, F, a) exp( -(E_DB - Z*e*F*(a'/2)) / (E_th + (hbar^2 * norm(q * kvector * (a'/norm(a)*Zsign))^2 / (2 * m * m_0))) );

% Sum over all nearest-neighbor sites (6 directions)
Z_h = 1;
for i = 1:6
    Z_h = Z_h + B_fac(E_DB, F, aa(i,:));
end

P = zeros(6,1);
for i = 1:6
    P(i) = 1/Z_h * B_fac(E_DB, F, aa(i,:));
end
end