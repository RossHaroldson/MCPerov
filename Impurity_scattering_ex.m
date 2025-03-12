% Impurity scattering
%% clear all
clear all
close all
clc
%% Constants
im= sqrt(-1);
%     Physical constants (SI units)
eps_0 = 8.854187818e-12; % vacuum dielctric constant (Farad/m)
m_0   = 9.109534e-31;    % free electron mass (kg)
hbar = 1.0545887e-34;   % Planck constant/(2pi) (J*s)
h=hbar*2*pi;
kb   = 1.380662e-23;    % Boltzmann constant (J/K)
%kb   = 8.6173303e-5;    % Boltzmann constant (eV/K)
e    = 1.6021892e-19;   % Proton charge (C)
%q = 1; % for eV units
%hbar = 6.582119514e-16; % reduced planck constant (eV*s)
a0   = 5.2917706e-11;   % Bohr radius (m)
a = 6.017e-10;          % lattice constant for cubic CsPbBr3 in meters
disp('Constants loaded');
%% Params
T=77;
Egap = 2.1*e;
Ef = 1*e;
a=6.3e-10;                                          % lattice constant for cspbbr3
m_e = 0.149;                                        % mass of electrons
m_h = 0.143;                                        % mass of hole
eps_rel_inf = 4.3;                                  % eps inf
eps_rel_stat = 32.3;                                % eps static for MAPbBr3 by M. Sendner 2016
Vcell = a^3;                                        % lattice volume
%functions
Ekin= @(k,m) hbar^2 .*norm(k)^2 ./2./(m.*m_0);
gradE = @(k,m) hbar^2 *norm(k)/(m*m_0);
knorm = @(E,m) sqrt(E*2*(m*m_0))/hbar;
v_k = @(k,m) 1/hbar * gradE(k,m);
vhat = @(k,m) 1/hbar/m/m_0 * k/norm(k);
Nqv = @(w) (exp(hbar*w/kb/T)-1)^-1;
Fermi = @(k,q,m,Ef,Egap) 1/(1-exp((Ekin(k+q,m)-Ef+Egap)/kb/T)); %probability k+q electron is occupied when fermi level is referenced from valanceband =0;
% for integral
n_den_e=1e22;       %carrier density in 1/m^3
q_scr=sqrt(e^2*n_den_e/(kb*T*eps_rel_inf*eps_0));       % screening wave vector or beta
Vbias = 3; %3 volts applied bias
t = 100e-9; % 100nm thickness of device
F = [Vbias/t/eps_rel_stat 0 0]; % electric field strength vector [Fx Fy Fz]
lat = [a 0 0; -a 0 0; 0 a 0; 0 -a 0; 0 0 a; 0 0 -a]; % nearest neighbor lattice vectors relative to defect
E_DB = 0.1*e;   %Diffusion barrier energy in Joules
E_th = kb*T;    %Thermal energy in joules
disp('Parameters loaded');
%% Brooks Herring Model
% with a potential of V(r) = Z*e/(4*pi*eps_stat*r)
% 1/tau(k) = 2*Z^2e^4*m*n_D*k/(pi*hbar^3*eps_stat^2*beta^2*(beta^2+4*k^2))
Z_imp = -1; % oxidation state of the ionized defect (-2,-1,0,1,2)
carrier = -1; % 1 for hole and -1 for electron
P_BH = @(k,Z,m_eff,n_D,beta) 2*Z^2*e^4*m_eff*m_0*n_D*norm(k)/(pi*hbar^3*eps_rel_stat^2*beta^2*(beta^2+4*norm(k)^2));
kx=linspace(0,knorm(0.3*e,m_e),100)';
ky=zeros(100,1);
kz=zeros(100,1);
kvector=[kx ky kz];
n_D=[1e18 1e19 1e20 1e21 1e22 1e23];
figure;hold on;
for j = 1:length(n_D)
    for i = 1:size(kvector,1)
        scat_BH(i,j) = P_BH(kvector(i,:),1,m_e,n_D(j),q_scr);
        E_e(i) = Ekin(kvector(i,:),m_e);
        % calculate the probablity a impurity will hop in 6 direction for
        % each k vector and impurit density. Phonons are with thermal
        % energy kT, k vector, and mass of electron.
        P_H_phonons(:,i)    = Prob_Hop(E_DB,F,lat,E_th,0,0,Z_imp,carrier); % for phonon causing a hop
        P_H_carriers(:,i)   = Prob_Hop(E_DB,F,lat,0,kvector(i,:),m_e,Z_imp,carrier); % for a electron or hole directly hitting an impurity to cause a hop
        P_H_3body(:,i)      = Prob_Hop(E_DB,F,lat,E_th,kvector(i,:),m_e,Z_imp,carrier);% for a electron/hole plus a phonon to cause an impurity to hop
    end
    plot(E_e/e,scat_BH(:,j),'linewidth',2);
end
hold off;set(gca, 'YScale', 'log')
title('Electron-Impurity Scattering Rates with screening'); xlabel('Energy of electron (eV)');
ylabel('Scattering Rate (seconds^{-1})');
% plot hop probabilities
 figure; plot(E_e/e, P_H_carriers,'linewidth',2);xlabel('Energy of carrier (eV)');ylabel('Probability of a impurity hop');title('Carriers');
 figure; plot(E_e/e, P_H_phonons,'linewidth',2);xlabel('Energy of carrier (eV)');ylabel('Probability of a impurity hop');title('Phonons');
 figure; plot(E_e/e, P_H_3body,'linewidth',2);xlabel('Energy of carrier (eV)');ylabel('Probability of a impurity hop');title('Phonon + Carrier Hop');
% figure; plot(E_e/e, P_H_carriers,E_e/e, P_H_phonons,E_e/e, P_H_3body,'linewidth',2)
% title('Hopping probabilities models'); xlabel('Energy of carrier (eV)');
% ylabel('Probability of a impurity hop');legend('Carriers','Phonons','3-Body')
% the hopping rate doesn't seem to be as directional as one would think.
% maybe the denominator that includes the carriers k vector should include
% a dot product to the unit lattice vectors. However, 
%% Scattering rates of BH + hopping models
% for every impurity density
%phonon_mode_energy = 4.42*0.001*e;
phonon_mode_energy = 19.42*0.001*e;
debye_freq = phonon_mode_energy/h;
for j = 1:length(n_D)
    for i = 1:size(kvector,1)
        % the max is to find the most hoppable direction
        hop_BH_therm_eq(i,j)= scat_BH(i,j)*max(P_H_phonons(:,i));
        hop_BH_carriers(i,j)= scat_BH(i,j)*max(P_H_carriers(:,i));
        hop_BH_3body(i,j)= scat_BH(i,j)*max(P_H_3body(:,i));
        % this is the rate at which a single impurity will hop
        hop_debye_phonons(i)= debye_freq*max(P_H_phonons(:,i));
    end
end
figure; semilogy(E_e/e, hop_BH_therm_eq,'linewidth',2)
title('Hopping rate from one electron with thermal model'); xlabel('Energy of carrier (eV)');
ylabel('Hopping Rate (1/s)');
figure; semilogy(E_e/e, hop_BH_carriers,'linewidth',2)
title('Hopping rates from one electron with carrier energy model'); xlabel('Energy of carrier (eV)');
ylabel('Hopping Rate (1/s)'); ylim([1e-16 max(max(hop_BH_carriers))]);
figure; semilogy(E_e/e, hop_BH_3body,'linewidth',2)
title('Hopping rates from one electron with 3 body energy hopping model'); xlabel('Energy of carrier (eV)');
ylabel('Hopping Rate (1/s)');
figure; semilogy(E_e/e, hop_debye_phonons,'linewidth',2)
title('Hopping rates with debye-phonon model no electron scattering'); xlabel('Energy of carrier (eV)');
ylabel('Hopping Rate (1/s)');
% Note that the hopping rate by the normal debye phonon model is 10^11 Hz
% while the scatter-hop rates are at 10^-11 1/s
% the scattering rate is the rate at which a electron will hit a impurity
% not the rate at which the impurity is hit by a electron

%% Solve for new k' vector
% Given an initial k vector with energy Ekin solve for the k' vector
% 
Ek = 0.01;
n_Ek = 70;
kout=[]; theta_BH = [];
%kz=linspace(0,knorm(0.3*e,m_e),1000)';
kx=-knorm(Ek*e,m_e)*ones(n_Ek,1);
kz=zeros(n_Ek,1);
ky=zeros(n_Ek,1);
kvector=[kx ky kz];
x = zeros(n_Ek,1); y = zeros(n_Ek,1); z = zeros(n_Ek,1);
m_eff = m_0*m_e;
Beta = sqrt(e^2*n_den_e/(kb*T*eps_rel_inf*eps_0));
for i = 1:length(kz)
    kout(:,i) = BH_kout(kvector(i,:)',m_eff,Beta);
    theta_BH(i) = 180/pi*dot(kout(:,i)/norm(kout(:,i)),kvector(i,:)'/norm(kvector(i,:)'));
end
% plots
figure; hold on;
quiver3(x',y',z',kout(1,:),kout(2,:),kout(3,:));
quiver3(0,0,0,kvector(1,1),kvector(1,2),kvector(1,3),'linewidth',2);
hold off;
xlabel('kx');ylabel('ky');zlabel('kz');
title(append('Scattering vectors of carrier at ', num2str(Ek), 'eV' ));

%% check to see if rod rotation is correct
vinc=[-1;-1;-1];
align = [1; 2; 3];
R = Rod_Rotation(vinc,align);
vincrot=R*vinc;
figure; hold on;
quiver3(0,0,0,align(1),align(2),align(3));
quiver3(0,0,0,vinc(1),vinc(2),vinc(3));
quiver3(0,0,0,vincrot(1),vincrot(2),vincrot(3));
hold off;
%% make a boxplot of scattering angles vs energy of carrier
%note that boxplot plots 1 box for each column of a matrix.
Emax = 0.3; % in eV
n_Ek = 100;
n_vary = 40;
kout=[]; theta_BH = zeros(n_Ek,n_vary); kvector =[];
kz = zeros(n_Ek,n_vary);
kx=kz;
ky=kz;
ks = linspace(0,knorm(Emax*e,m_e),n_vary);
energies = linspace(0,Emax,n_vary);
for i = 1:n_Ek
    kz(i,:)=ks;
end
%x = zeros(n_Ek,1); y = zeros(n_Ek,1); z = zeros(n_Ek,1);
m_eff = m_0*m_e;
Beta = sqrt(e^2*n_den_e/(kb*T*eps_rel_inf*eps_0));
for j = 1:n_vary
for i = 1:n_Ek
    kvector=[kx(i,j); ky(i,j); kz(i,j);];
    kout = BH_kout(kvector,m_eff,Beta);
    theta_BH(i,j) = 180/pi*dot(kout/norm(kout),kvector/norm(kvector));
end
end
% plots
figure; boxplot(theta_BH,'Labels',energies,'Orientation','horizontal');
xlabel('Scattering angle (degrees)');ylabel('Carrier Energy (eV)');
title('Boxplot statistics of scattering angles vs electron energy');