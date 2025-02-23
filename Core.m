%% Core of model
clear; clc;
%% Constants
im = sqrt(-1);
eps_0 = 8.854187818e-12;  % vacuum permittivity (F/m)
m_0 = 9.109534e-31;       % free electron mass (kg)
hbar = 1.0545887e-34;      % Planck constant/(2pi) (J*s)
h = 2*pi*hbar;
kb = 1.380662e-23;        % Boltzmann constant (J/K)
q = 1.6021892e-19;        % Proton charge (C)
a0 = 5.2917706e-11;        % Bohr radius (m)
disp('Constants loaded');
%% Parameters
T = 300;  
a = 6.017e-10;            % lattice constant for cubic CsPbBr3 (m)
N_0 = 1/a^3;              % lattice concentration (m^-3)
density = 4420;           % density (kg/m^3)
Ntime = 3000;             % number of time steps
N_e = 10000;              % number of electrons
m_e = 0.149;              % effective electron mass
N_h = 10000;              % number of holes
m_h = 0.143;              % effective hole mass
Econd = -2.93;            
Eval = -5.3;              
DFE_Ai = 1.03*q;          
N_Ai = 100;               
E_Ai = (-5.2 - Eval)*q;     
DFE_Va = 1.1*q;           
N_Va = 100;               
E_Va = (-3.03 - Eval)*q;    
nonpar_e = 1.033;
nonpar_h = 1.054;
Egap = (Econd - Eval)*q;  
device_thickness = 1000e-10;
Ef = Egap/2 + 3/4*kb*T*log(m_h/m_e);
total_Ai_density = N_0*exp(-DFE_Ai/(kb*T));
total_Va_density = N_0*exp(-DFE_Va/(kb*T));
[total_e_density, total_h_density, ~, ~, ~] = fermifinder2(m_e, m_h, total_Ai_density, E_Ai, 1, total_Va_density, 1, E_Va, Egap, T);
stat_weight_e = device_thickness*total_e_density/N_e;
stat_weight_h = device_thickness*total_h_density/N_h;
eps_rel_inf = 4.3;
w_TO_1 = 4.83e-3*q;
w_TO_2 = 11.66e-3*q;
w_LO_1 = 5.62e-3*q;
w_LO_2 = 19.42e-3*q;
d_e = -2.93*q;
d_h = -2.2*q;
C = 2.1e10;
C_MAPbBr3_11 = 3.13e10;
C_MAPbBr3_44 = 4.3e9;
eps = eps_rel_inf*eps_0;
npoints = 1000;
realspace = linspace(0, 1000e-10, npoints);
dx = realspace(2) - realspace(1);
ITO_workfunction = -4.7;
InGa_workfunction = -4.2;
Vbias = 1;
del_t_eh = 5e-17;
del_t_ions = 1e-15;
kmaxnorm = dx/del_t_eh * m_e*m_0/hbar;
Emax = 0.42e-18;
Ekin = @(k, m) hbar^2*norm(k)^2/(2*m*m_0);
gradE = @(k, m) hbar^2*norm(k)/(m*m_0);
knorm = @(E, m) sqrt(2*m*m_0*E)/hbar;
v_k = @(k, m) gradE(k, m)/hbar;
disp('Parameters loaded');
%% Initialize Electron and Hole Distributions
r_e = rand(N_e,1) * device_thickness;
k_e = rand(N_e,3)*2-1; 
k_e = (k_e' ./ vecnorm(k_e'))';  % normalize rows
r_h = rand(N_h,1) * device_thickness;
k_h = rand(N_h,3)*2-1; 
k_h = (k_h' ./ vecnorm(k_h'))';

Ep = exp(linspace(log(Egap), log(Egap+Emax), 100000));
Ep(1) = Egap;
prob_dist_e = 8*pi*sqrt(2)/(hbar*2*pi)^3*(m_e*m_0)^(3/2)*sqrt(Ep-Egap)./(1+exp((Ep-Ef)/(kb*T)));
prob_dist_h = 8*pi*sqrt(2)/(hbar*2*pi)^3*(m_h*m_0)^(3/2)*sqrt(Ep-Egap)./(1+exp((Ep-Ef)/(kb*T)));
figure; 
plot(Ep/q - Egap/q, prob_dist_e, Ep/q - Egap/q, prob_dist_h);
xlim([Ep(1)/q - Egap/q, Emax/q - Egap/q]);
Epart_e = randpdf(prob_dist_e, Ep, [1 N_e]);
Epart_h = randpdf(prob_dist_h, Ep, [1 N_h]);
for i = 1:length(k_e)
    k_e(i,:) = k_e(i,:) * knorm(Epart_e(i)-Egap, m_e);
end
for i = 1:length(k_h)
    k_h(i,:) = k_h(i,:) * knorm(Epart_h(i)-Egap, m_h);
end
energy = linspace(Egap, Egap+Emax, 1000);
electron_Ekin = arrayfun(@(i) Ekin(k_e(i,:), m_e), 1:length(k_e))';
hole_Ekin = arrayfun(@(i) Ekin(k_h(i,:), m_h), 1:length(k_h))';
Eedensity = cloudincell(electron_Ekin+Egap, energy);
Ehdensity = cloudincell(hole_Ekin+Egap, energy);
figure;
plot(energy./q, Eedensity, energy./q, Ehdensity);
xlabel('Kinetic Energy (eV)'); ylabel('Density per unit energy');
title('Carrier Probability Distribution');
legend('Electrons','Holes');
disp('Particles Initialized');

%% Compute charge density and RHS (note: stat_weight_e and stat_weight_h are assumed from Load_Params)
rho = cloudincell(r_h,realspace)*q*stat_weight_h - cloudincell(r_e,realspace)*q*stat_weight_e;
RHS = q/eps * rho;
disp(['Avg electron count per bin: ' num2str(mean(histcounts(r_e,realspace)))]);

figure; 
plot((cloudincell(r_e,realspace) - cloudincell(r_h,realspace))*dx);
title('Cloud in cell difference');

% Set up Poisson matrix (using vectorized diagonal construction)
n = numel(realspace); 
alpha = 1/dx^2;
poismatrix = 2*alpha*eye(n) - alpha*diag(ones(n-1,1),1) - alpha*diag(ones(n-1,1),-1);
bx = RHS;
bx(end) = bx(end) + alpha*Vbias*q;

% Solve Poisson equation for potential and compute electric field
V = poismatrix \ bx';
psi = V/q;
Efield = Efield_solver(psi,dx);
figure; plot(realspace*1e9, Efield);
title('Initial Electric Field at t=0');
xlabel('Position (nm)'); ylabel('Electric Field (V/m)');
disp('Initial potential and field solved');

% Define scattering rates (acoustic scattering for electrons and holes)
P_e_ac = @(Ekin) d_e^2 * sqrt(2)*(m_e*m_0)^(3/2)*kb*T*sqrt(Ekin) / (pi*hbar^4*C);
P_h_ac = @(Ekin) d_h^2 * sqrt(2)*(m_h*m_0)^(3/2)*kb*T*sqrt(Ekin) / (pi*hbar^4*C);
ac_scat_scale = 1e2;
scat_rate_e_ac = ac_scat_scale * P_e_ac(linspace(0,0.3*q,1000));
scat_rate_h_ac = ac_scat_scale * P_h_ac(linspace(0,0.3*q,1000));
figure; 
plot(linspace(0,0.3,1000), scat_rate_e_ac, linspace(0,0.3,1000), scat_rate_h_ac);
title('Acoustic scattering rate');
xlabel('Energy (eV)'); ylabel('Scattering rate (s^{-1})');
legend('Electrons','Holes');

%% Preallocate observables and time-dependent variables
J = zeros(1,Ntime);
charge_collected_ITO = zeros(1,Ntime);
charge_collected_Cu = zeros(1,Ntime);
Avg_Vel_e = zeros(1,Ntime);
Avg_Vel_h = zeros(1,Ntime);
Psi = zeros(n,Ntime);
elec_vs_t = zeros(n-1,Ntime);
hole_vs_t = zeros(n-1,Ntime);
charge_injected_ITO = zeros(1,Ntime);
charge_injected_Cu = zeros(1,Ntime);
e_parts = N_e/npoints;
h_parts = N_h/npoints;
stat_weight = 1;

% Vectorized kinetic energy function (for scattering rates)
Ekin_vec = @(k, m) hbar^2 * sum(k.^2,2) / (2*m*m_0);

%profile on;
for t = 1:Ntime
    % Check time step criteria for electrons and holes
    if abs(hbar*max(abs(k_e(:,3)))/(m_e*m_0)*del_t_eh + q*Vbias*del_t_eh^2/(hbar*device_thickness)) > dx
        disp(['Electron dt too high; recommended dt: ' num2str(dx/(max(abs(k_e(:,3)))*m_e*m_0/hbar))]);
    elseif abs(hbar*max(abs(k_h(:,3)))/(m_h*m_0)*del_t_eh + q*Vbias*del_t_eh^2/(hbar*device_thickness)) > dx
        disp(['Hole dt too high; recommended dt: ' num2str(dx/(max(abs(k_h(:,3)))*m_h*m_0/hbar))]);
    end
    
    % Update potential based on current charge density
    rho = (cloudincell_gpu2(r_h,realspace) - cloudincell_gpu2(r_e,realspace)) * q * stat_weight;
    V = poisson_solver(poismatrix, q/eps * rho, Vbias);
    psi = V/q;
    Efield = Efield_solver(psi,dx);
    
    % Preallocate GPU variables and compute scattering rates vectorized
    gpu_r_e = gpuArray(r_e); 
    xe = gpuArray(k_e(:,1)); ye = gpuArray(k_e(:,2)); ze = gpuArray(k_e(:,3));
    gpu_r_h = gpuArray(r_h); 
    xh = gpuArray(k_h(:,1)); yh = gpuArray(k_h(:,2)); zh = gpuArray(k_h(:,3));
    
    scatter_e = P_e_ac(Ekin_vec(k_e, m_e));
    scatter_h = P_h_ac(Ekin_vec(k_h, m_h));
    gpu_Efield_int_e = gpuArray(interp1(realspace, Efield, r_e));
    gpu_Efield_int_h = gpuArray(interp1(realspace, Efield, r_h));
    gpu_scatter_e = gpuArray(scatter_e); 
    gpu_scatter_h = gpuArray(scatter_h);
    gpu_rand_e = rand(length(r_e),1,'gpuArray');
    gpu_rand_h = rand(length(r_h),1,'gpuArray');
    
    % Run particle loops on the GPU (arrayfun calls unchanged)
    [r_e_new, xe_new, ye_new, ze_new, elec_into_Cu, elec_into_ITO] = ...
        arrayfun(@electronloopV1, gpu_r_e, xe, ye, ze, m_e, del_t_eh, gpu_scatter_e, device_thickness, gpu_Efield_int_e, gpu_rand_e);
    r_e = gather(r_e_new); 
    k_e = [gather(xe_new) gather(ye_new) gather(ze_new)];
    
    [r_h_new, xh_new, yh_new, zh_new, hole_into_Cu, hole_into_ITO] = ...
        arrayfun(@holeloopV1, gpu_r_h, xh, yh, zh, m_h, del_t_eh, gpu_scatter_h, device_thickness, gpu_Efield_int_h, gpu_rand_h);
    r_h = gather(r_h_new); 
    k_h = [gather(xh_new) gather(yh_new) gather(zh_new)];
    
    % Remove out-of-bound particles
    r_e = r_e(~isnan(r_e)); 
    r_h = r_h(~isnan(r_h));
    k_e(any(isnan(k_e),2),:) = []; 
    k_h(any(isnan(k_h),2),:) = [];
    
    % Charge injection at contacts (using the bucket helper function)
    LDh = h_parts - bucket(r_h, [realspace(1) realspace(2)]);
    LDe = e_parts - bucket(r_e, [realspace(1) realspace(2)]);
    RDh = h_parts - bucket(r_h, [realspace(end-1) realspace(end)]);
    RDe = e_parts - bucket(r_e, [realspace(end-1) realspace(end)]);
    
    % Injection routines (the injection functions are compacted here; note that the randomized k_dum is computed in one line)
    if LDe > 0
        LDi = (length(r_e)+1):(length(r_e)+LDe);
        k_dum = (rand(LDe,3)*2-1);
        for i = 1:size(k_dum,1)
            k_dum(i,:) = k_dum(i,:) * knorm(Epart_e(randperm(length(Epart_e),1))-Egap, m_e);
        end
        k_e(LDi,:) = k_dum;
        r_e(LDi) = hbar * k_e(LDi,3) / (m_e*m_0) * del_t_eh;
    end
    if LDh > 0
        LDi = (length(r_h)+1):(length(r_h)+LDh);
        k_dum = (rand(LDh,3)*2-1);
        for i = 1:size(k_dum,1)
            k_dum(i,:) = k_dum(i,:) * knorm(Epart_h(randperm(length(Epart_h),1))-Egap, m_h);
        end
        k_h(LDi,:) = k_dum;
        r_h(LDi) = hbar * k_h(LDi,3) / (m_h*m_0) * del_t_eh;
    end
    if RDh > 0
        RDi = (length(r_h)+1):(length(r_h)+RDh);
        k_dum = (rand(RDh,3)*2-1); k_dum(:,3) = -abs(k_dum(:,3));
        for i = 1:size(k_dum,1)
            k_dum(i,:) = k_dum(i,:) * knorm(Epart_h(randperm(length(Epart_h),1))-Egap, m_h);
        end
        k_h(RDi,:) = k_dum;
        r_h(RDi) = device_thickness + hbar * k_h(RDi,3) / (m_h*m_0) * del_t_eh;
    end
    if RDe > 0
        RDi = (length(r_e)+1):(length(r_e)+RDe);
        k_dum = (rand(RDe,3)*2-1); k_dum(:,3) = -abs(k_dum(:,3));
        for i = 1:size(k_dum,1)
            k_dum(i,:) = k_dum(i,:) * knorm(Epart_e(randperm(length(Epart_e),1))-Egap, m_e);
        end
        k_e(RDi,:) = k_dum;
        r_e(RDi) = device_thickness + hbar * k_e(RDi,3) / (m_e*m_0) * del_t_eh;
    end

    % Update observables
    charge_collected_ITO(t) = sum(gather(hole_into_ITO)) - sum(gather(elec_into_ITO));
    charge_collected_Cu(t) = sum(gather(hole_into_Cu)) - sum(gather(elec_into_Cu));
    charge_injected_ITO(t) = LDh - LDe;
    charge_injected_Cu(t) = RDh - RDe;
    J(t) = (charge_collected_Cu(t) - charge_injected_Cu(t) + charge_injected_ITO(t) - charge_collected_ITO(t)) * q * stat_weight / del_t_eh;
    Avg_Vel_e(t) = hbar * mean(k_e(:,3)) / (m_e*m_0);
    Avg_Vel_h(t) = hbar * mean(k_h(:,3)) / (m_h*m_0);
    Psi(:,t) = V;
    elec_vs_t(:,t) = histcounts(r_e,realspace);
    hole_vs_t(:,t) = histcounts(r_h,realspace);
    disp(t);
end
%profile viewer;

%% Compute overall observables and quasi-Fermi levels
J_obs = q * stat_weight * (sum(hole_vs_t,1).*Avg_Vel_h - sum(elec_vs_t,1).*Avg_Vel_e);
disp(['Time steps per second: ' num2str(t/toc)]);

Nquasi = 200;
e_density = cloudincell(r_e, linspace(0,1000e-10,Nquasi)) * stat_weight;
h_density = cloudincell(r_h, linspace(0,1000e-10,Nquasi)) * stat_weight;
quasi_Efn = zeros(1,Nquasi); quasi_Efp = zeros(1,Nquasi);
parfor i = 1:length(e_density)
    [qFn, qFp] = quasifermifind(e_density(i), h_density(i), m_e, m_h, Egap);
    quasi_Efn(i) = qFn; 
    quasi_Efp(i) = qFp;
end
figure;
plot(linspace(0,100,Nquasi), quasi_Efn/q, linspace(0,100,Nquasi), quasi_Efp/q, linspace(0,100,Nquasi), Ef/q*ones(1,Nquasi));
xlabel('Device position (nm)'); ylabel('Energy (eV)');
legend('Electrons','Holes','Intrinsic');

%% (Additional post-processing plots can be similarly condensed.)
%%%
figure;
plot(linspace(0,t*del_t_eh,t),smooth(smooth(J(1:t))),linspace(0,t*del_t_eh,t),J_obs(1:t));
grid on
title('Current vs Time')
xlabel('Time (s)')
ylabel('Current (A/m^2)') % 1A/m^2 is 0.1mA/cm^2
%%%
figure;
plot(linspace(0,t*del_t_eh,t),sum(elec_vs_t(:,1:t)),linspace(0,t*del_t_eh,t),sum(hole_vs_t(:,1:t)))
grid on
title('Number of Charges vs Time')
xlabel('Time (s)')
ylabel('Carrier particles')
legend({'Electrons','Holes'});
%%%
figure;
%plot(linspace(0,t*del_t_eh,t),smooth(charge_collected_Cu),linspace(0,t*del_t_eh,t),smooth(charge_collected_ITO),linspace(0,t*del_t_eh,t),smooth(charge_injected_Cu),linspace(0,t*del_t_eh,t),smooth(charge_injected_ITO))
plot(linspace(0,t*del_t_eh,t),charge_collected_Cu(1:t),linspace(0,t*del_t_eh,t),charge_collected_ITO(1:t),linspace(0,t*del_t_eh,t),charge_injected_Cu(1:t),linspace(0,t*del_t_eh,t),charge_injected_ITO(1:t))
grid on
title('Number of Charges injected  by contacts vs Time')
xlabel('Time (s)')
ylabel('Carrier particles')
legend({'Charge collected at Cu','Charge collected at ITO','Charge injected at Cu','Charge injected at ITO'});
%%%
figure;
%plot(linspace(0,t*del_t_eh,t),smooth(charge_collected_Cu),linspace(0,t*del_t_eh,t),smooth(charge_collected_ITO),linspace(0,t*del_t_eh,t),smooth(charge_injected_Cu),linspace(0,t*del_t_eh,t),smooth(charge_injected_ITO))
charge_inout=charge_injected_ITO+charge_injected_Cu-charge_collected_Cu-charge_collected_ITO;
plot(linspace(0,t*del_t_eh,t),charge_inout(1:t));
grid on
title('Number of Charges passing through contacts vs Time')
xlabel('Time (s)')
ylabel('Carrier particles')
%legend({'Charge collected at Cu','Charge collected at ITO','Charge injected at Cu','Charge injected at ITO'});
sum(charge_inout)
%%%
figure;
mesh(hole_vs_t);view(2);
title('Hole Particle Density vs Time')
xlabel('Time step')
ylabel('Position in device (Angstrom)')
colorbar
%%%
figure;
mesh(elec_vs_t);view(2);
title('Electron Particle Density vs Time')
xlabel('Time step')
ylabel('Position in device (Angstrom)')
colorbar
