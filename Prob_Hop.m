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

%     Physical constants (SI units)
im = sqrt(-1);
%     Physical constants (SI units)
eps_0 = 8.854187818e-12; % vacuum dielctric constant (Farad/m)
m_0   = 9.109534e-31;    % free electron mass (kg)
hbar = 1.0545887e-34;   % Planck constant/(2pi) (J*s)
h = hbar*2*pi;
kb   = 1.380662e-23;    % Boltzmann constant (J/K)
a0   = 5.2917706e-11;   % Bohr radius (m)
e = 1.6021892e-19;   % Proton charge (C)
if m==0
    m=1;
end
if Z ~= 0
    Zsign=Z/norm(Z); % provides the sign of the ionized defect
else
    Zsign=Z; % keeps from dividing by zero
end

%   function handles
B_fac = @(E_DB,F,a) exp(-(E_DB-Z*e*F*a'/2)/(E_th+(hbar^2 .*norm(q*kvector*(a'/norm(a)*Zsign))^2 ./2./(m.*m_0))));
Z_h = 1; % the boltzmann factor for the defect to remain stationary is set to 1 since it is the ground state.
for i= 1:6
    % sum over all Boltzmann factors
    Z_h = Z_h + B_fac(E_DB,F,aa(i,:));
end
P=zeros(6,1);
for i=1:6
    % Calculate the probablity of each a vector
    P(i) = 1/Z_h * B_fac(E_DB,F,aa(i,:));
end
end