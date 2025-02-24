%% Core_refactored.m
clear; clc;

%% Load Constants & Parameters
params = initParameters();
disp('Constants & Parameters loaded');

%% Initialize Particles & Energy Distributions
[particles, energyStruct] = initParticles(params);
disp('Particles Initialized');

%% Initial Density, Potential, and Field Computation
density_e = cloudincell(particles.r_e, params.realspace) * params.q * params.stat_weight_e;
density_h = cloudincell(particles.r_h, params.realspace) * params.q * params.stat_weight_h;
rho = density_h - density_e;
RHS = params.q/params.eps * rho;

[poismatrix, alpha] = buildPoissonMatrix(params.dx, length(params.realspace));
bx = RHS;
bx(end) = bx(end) + alpha*params.Vbias*params.q;
V = poismatrix \ bx';
psi = V./params.q;
Efield = Efield_solver(psi, params.dx);
t = 1;
disp('Initial potential and field solved');

plotInitialFields(params, Efield, particles);

%% Preallocate Observables
observables = initObservables(params, length(params.realspace));

tic;
while t <= params.Ntime
    checkDt(particles, params);
    
    % Update potential & field using GPU-based density estimation
    rho = (cloudincell_gpu2(particles.r_h, params.realspace) - cloudincell_gpu2(particles.r_e, params.realspace)) * params.q;
    V = poisson_solver(poismatrix, params.q/params.eps * rho, params.Vbias);
    observables.Psi(:,t) = V/params.q;
    observables.Efield(:,t) = Efield_solver(observables.Psi(:,t), params.dx);
    
    % Update scattering rates and prepare GPU arrays
    [particles, scatRates, gpu_Efield_e, gpu_Efield_h, gpu_rand_e, gpu_rand_h] = updateScattering(particles, params, Efield);
    
    % Update particle positions via unified GPU loop
    [particles.r_e, particles.k_e, into.elec_into_Cu, into.elec_into_ITO] = updateParticles_gpu('electron', particles.r_e, particles.k_e, params, gpu_Efield_e, gpu_rand_e, scatRates.e);
    [particles.r_h, particles.k_h, into.hole_into_Cu, into.hole_into_ITO] = updateParticles_gpu('hole', particles.r_h, particles.k_h, params, gpu_Efield_h, gpu_rand_h, scatRates.h);
    
    % Clean up and inject new particles
    [particles.r_e, particles.k_e] = removeNaNParticles(particles.r_e, particles.k_e);
    [particles.r_h, particles.k_h] = removeNaNParticles(particles.r_h, particles.k_h);
    [particles.r_e, particles.k_e, injLefte, injRighte] = injectParticles(particles.r_e, particles.k_e, params, energyStruct.Epart_e, 'electron');
    [particles.r_h, particles.k_h, injLefth, injRighth] = injectParticles(particles.r_h, particles.k_h, params, energyStruct.Epart_h, 'hole');
    injLeft=injLefth -injLefte;
    injRight=injRighth -injRighte;

    % Update observables
    observables = updateObservables(observables, into, injLeft, injRight, particles, params, t);
    
    if mod(t,100)==0
        disp("Plotting")
        updateLivePlots(observables, params, t);
        drawnow;
        disp(['Time step: ', num2str(t)]);
    end
    t=t+1;
end
fprintf('Time steps per second: %.2f\n', params.Ntime/toc);

%% Compute Quasi-Fermi Levels
computeQuasiFermiLevels(particles, params);
%% Post-Processing Plots
plotPostProcessing(observables, params);
%% --- Local Helper Functions ---

function params = initParameters()
    % Physical constants
    params.eps_0 = 8.854187818e-12;   params.m_0   = 9.109534e-31;
    params.hbar  = 1.0545887e-34;      params.h     = 2*pi*params.hbar;
    params.kb    = 1.380662e-23;        params.q     = 1.6021892e-19;
    params.a0    = 5.2917706e-11;
    
    % Simulation parameters
    params.T           = 300;    params.a       = 6.017e-10;
    params.N_0         = 1/params.a^3;  params.density = 4420;
    params.Ntime       = 3000;   params.N_e     = 10000;
    params.m_e         = 0.149;  params.N_h     = 10000;
    params.m_h         = 0.143;  params.Econd   = -2.93;
    params.Eval        = -5.3;   params.DFE_Ai  = 1.03*params.q;
    params.N_Ai        = 100;    params.E_Ai    = (-5.2 - params.Eval)*params.q;
    params.DFE_Va      = 1.1*params.q;  params.N_Va = 100;
    params.E_Va        = (-3.03 - params.Eval)*params.q;
    params.nonpar_e    = 1.033;  params.nonpar_h = 1.054;
    params.Egap        = (params.Econd - params.Eval)*params.q;
    params.device_thickness = 1000e-10;
    params.Ef = params.Egap/2 + 3/4*params.kb*params.T*log(params.m_h/params.m_e);
    
    % Densities (via fermifinder2, assumed available)
    total_Ai_density = params.N_0 * exp(-params.DFE_Ai/(params.kb*params.T));
    total_Va_density = params.N_0 * exp(-params.DFE_Va/(params.kb*params.T));
    [total_e_density, total_h_density] = fermifinder2(params.m_e, params.m_h, total_Ai_density, params.E_Ai, 1, total_Va_density, 1, params.E_Va, params.Egap, params.T);
    params.stat_weight_e = params.device_thickness * total_e_density / params.N_e;
    params.stat_weight_h = params.device_thickness * total_h_density / params.N_h;
    
    % Additional parameters
    params.eps_rel_inf   = 4.3;   params.w_TO_1 = 4.83e-3*params.q;
    params.w_TO_2        = 11.66e-3*params.q;   params.w_LO_1 = 5.62e-3*params.q;
    params.w_LO_2        = 19.42e-3*params.q;   params.d_e = -2.93*params.q;
    params.d_h           = -2.2*params.q;       params.C   = 2.1e10;
    params.C_MAPbBr3_11  = 3.13e10;             params.C_MAPbBr3_44 = 4.3e9;
    params.eps = params.eps_rel_inf * params.eps_0;
    
    % Spatial grid & bias
    params.npoints   = 1000;
    params.realspace = linspace(0, params.device_thickness, params.npoints);
    params.dx        = diff(params.realspace(1:2));
    params.ITO_workfunction = -4.7;
    params.InGa_workfunction = -4.2;
    params.Vbias = 1;
    
    % Time steps & scattering parameters
    params.del_t_eh   = 5e-17;    params.del_t_ions = 1e-15;
    params.kmaxnorm   = params.dx/params.del_t_eh * params.m_e*params.m_0/params.hbar;
    params.Emax       = 0.42e-18;
    params.Ekin       = @(k, m) params.hbar^2 * sum(k.^2, 2) / (2*m*params.m_0);
    params.gradE      = @(k, m) params.hbar^2 * sqrt(sum(k.^2, 2))/(m*params.m_0);
    params.knorm      = @(E, m) sqrt(2*m*params.m_0*E)/params.hbar;
    params.v_k        = @(k, m) params.gradE(k, m)/params.hbar;
    
    % Acoustic scattering functions
    params.P_e_ac = @(Ekin) params.d_e^2 * sqrt(2)*(params.m_e*params.m_0)^(3/2) * params.kb * params.T * sqrt(Ekin) / (pi*params.hbar^4*params.C);
    params.P_h_ac = @(Ekin) params.d_h^2 * sqrt(2)*(params.m_h*params.m_0)^(3/2) * params.kb * params.T * sqrt(Ekin) / (pi*params.hbar^4*params.C);
    params.ac_scat_scale = 1e2;
    
    % Energy distributions for carriers
    Ep = exp(linspace(log(params.Egap), log(params.Egap+params.Emax), 100000));
    Ep(1) = params.Egap;
    density_factor = 8*pi*sqrt(2)/(params.hbar*2*pi)^3;
    params.prob_dist_e = density_factor*(params.m_e*params.m_0)^(3/2)*sqrt(Ep-params.Egap) ./ (1+exp((Ep-params.Ef)/(params.kb*params.T)));
    params.prob_dist_h = density_factor*(params.m_h*params.m_0)^(3/2)*sqrt(Ep-params.Egap) ./ (1+exp((Ep-params.Ef)/(params.kb*params.T)));
    params.Ep = Ep;
end

function [particles, energyStruct] = initParticles(params)
    % Initialize positions and random k-vectors for electrons & holes
    particles.r_e = rand(params.N_e, 1) * params.device_thickness;
    particles.k_e = (rand(params.N_e, 3)*2 - 1);
    particles.k_e = particles.k_e ./ sqrt(sum(particles.k_e.^2,2));
    particles.r_h = rand(params.N_h, 1) * params.device_thickness;
    particles.k_h = (rand(params.N_h, 3)*2 - 1);
    particles.k_h = particles.k_h ./ sqrt(sum(particles.k_h.^2,2));
    
    % Sample energy partitions and update k-vectors accordingly
    energyStruct.Epart_e = randpdf(params.prob_dist_e, params.Ep, [1, params.N_e]);
    energyStruct.Epart_h = randpdf(params.prob_dist_h, params.Ep, [1, params.N_h]);
    for i = 1:params.N_e
        particles.k_e(i,:) = particles.k_e(i,:) * params.knorm(energyStruct.Epart_e(i)-params.Egap, params.m_e);
    end
    for i = 1:params.N_h
        particles.k_h(i,:) = particles.k_h(i,:) * params.knorm(energyStruct.Epart_h(i)-params.Egap, params.m_h);
    end
end

function [M, alpha] = buildPoissonMatrix(dx, n)
    alpha = 1/dx^2;
    M = 2*alpha*eye(n) - alpha*diag(ones(n-1,1),1) - alpha*diag(ones(n-1,1),-1);
end

function observables = initObservables(params, n)
    observables = struct(...
        'J', zeros(1, params.Ntime), ...
        'charge_collected_ITO', zeros(1, params.Ntime), ...
        'charge_collected_Cu', zeros(1, params.Ntime), ...
        'Avg_Vel_e', zeros(1, params.Ntime), ...
        'Avg_Vel_h', zeros(1, params.Ntime), ...
        'Psi', zeros(n, params.Ntime), ...
        'elec_vs_t', zeros(n-1, params.Ntime), ...
        'hole_vs_t', zeros(n-1, params.Ntime), ...
        'charge_injected_ITO', zeros(1, params.Ntime), ...
        'charge_injected_Cu', zeros(1, params.Ntime), ...
        'elec_into_Cu', zeros(1, params.Ntime), ...
        'hole_into_Cu', zeros(1, params.Ntime), ...
        'elec_into_ITO', zeros(1, params.Ntime), ...
        'hole_into_ITO', zeros(1, params.Ntime), ...
        'injLeft', zeros(1, params.Ntime), ...
        'injRight', zeros(1, params.Ntime), ...
        'Efield', zeros(n,params.Ntime) );
end

function checkDt(particles, params)
    max_ke = max(abs(particles.k_e(:,3)));
    dt_e = params.hbar*max_ke/(params.m_e*params.m_0)*params.del_t_eh + params.q*params.Vbias*params.del_t_eh^2/(params.hbar*params.device_thickness);
    if abs(dt_e) > params.dx
        disp(['Electron dt too high; recommended dt: ' num2str(params.dx/(max_ke*params.m_e*params.m_0/params.hbar))]);
    end
    max_kh = max(abs(particles.k_h(:,3)));
    dt_h = params.hbar*max_kh/(params.m_h*params.m_0)*params.del_t_eh + params.q*params.Vbias*params.del_t_eh^2/(params.hbar*params.device_thickness);
    if abs(dt_h) > params.dx
        disp(['Hole dt too high; recommended dt: ' num2str(params.dx/(max_kh*params.m_h*params.m_0/params.hbar))]);
    end
end

function [particles, scatRates, gpu_Efield_e, gpu_Efield_h, gpu_rand_e, gpu_rand_h] = updateScattering(particles, params, Efield)
    Ekin_e = params.Ekin(particles.k_e, params.m_e);
    Ekin_h = params.Ekin(particles.k_h, params.m_h);
    scatRates.e = params.ac_scat_scale * params.P_e_ac(Ekin_e);
    scatRates.h = params.ac_scat_scale * params.P_h_ac(Ekin_h);
    gpu_Efield_e = gpuArray(interp1(params.realspace, Efield, particles.r_e));
    gpu_Efield_h = gpuArray(interp1(params.realspace, Efield, particles.r_h));
    gpu_rand_e = rand(length(particles.r_e), 1, 'gpuArray');
    gpu_rand_h = rand(length(particles.r_h), 1, 'gpuArray');
end

function [r_new, k_new, into_Cu, into_ITO] = updateParticles_gpu(type, r, k, params, gpu_Efield, gpu_rand, scatter)
    if strcmp(type, 'electron')
        signVal = -1; m_val = params.m_e;
    else
        signVal = 1; m_val = params.m_h;
    end
    del_t_eh = params.del_t_eh;  device_thickness = params.device_thickness;
    [r_new, kx_new, ky_new, kz_new, into_Cu, into_ITO] = arrayfun(@carrierLoopV1, ...
        gpuArray(r), gpuArray(k(:,1)), gpuArray(k(:,2)), gpuArray(k(:,3)), ...
        m_val, del_t_eh, gpuArray(scatter), device_thickness, gpu_Efield, gpu_rand, signVal);
    r_new = gather(r_new);
    k_new = [gather(kx_new) gather(ky_new) gather(kz_new)];
end

function [r, kx, ky, kz, into_Cu, into_ITO] = carrierLoopV1(r_val, kx, ky, kz, m_val, dt, scatter, device_thickness, Efield_int, r_rand, signVal)
    m0 = 9.109534e-31; hbar = 1.0545887e-34; q = 1.6021892e-19;
    into_Cu = 0; into_ITO = 0;
    if dt * scatter > r_rand
        vecnorm = sqrt(kx^2 + ky^2 + kz^2);
        r1 = 2*rand() - 1; r2 = 2*rand() - 1; r3 = 2*rand() - 1;
        rnd_norm = sqrt(r1^2 + r2^2 + r3^2);  if rnd_norm == 0, rnd_norm = 1; end
        r1 = r1 / rnd_norm; r2 = r2 / rnd_norm; r3 = r3 / rnd_norm;
        kx = r1 * vecnorm;
        ky = r2 * vecnorm;
        kz = r3 * vecnorm + signVal*(q/hbar * Efield_int * dt);
        r = r_val + hbar*kz/(m_val*m0)*dt + signVal*(q/hbar * Efield_int * dt^2);
    else
        kz = kz + signVal*(q/hbar * Efield_int * dt);
        r = r_val + hbar*kz/(m_val*m0)*dt + signVal*(q/hbar * Efield_int * dt^2);
    end
    if r > device_thickness
         r = NaN; kx = NaN; ky = NaN; kz = NaN;  into_Cu = 1;
    elseif r < 0
         r = NaN; kx = NaN; ky = NaN; kz = NaN;  into_ITO = 1;
    end
end

function [r, k] = removeNaNParticles(r, k)
    valid = ~isnan(r);
    r = r(valid);  k = k(valid,:);
end

function [r, k, injLeft, injRight] = injectParticles(r, k, params, Epart, type)
    leftCount  = bucket(r, [params.realspace(1) params.realspace(2)]);
    rightCount = bucket(r, [params.realspace(end-1) params.realspace(end)]);
    if strcmp(type,'electron')
        desired = params.N_e/params.npoints; m_eff = params.m_e;
    else
        desired = params.N_h/params.npoints; m_eff = params.m_h;
    end
    injLeft  = max(0, desired - leftCount);
    injRight = max(0, desired - rightCount);
    if injLeft > 0
        [r, k] = injectBoundary(r, k, injLeft, Epart, params, m_eff, type, 'left');
    end
    if injRight > 0
        [r, k] = injectBoundary(r, k, injRight, Epart, params, m_eff, type, 'right');
    end
end

function [r, k] = injectBoundary(r, k, injCount, Epart, params, m_eff, type, side)
    k_new = rand(injCount, 3)*2 - 1;
    if strcmp(side, 'right')
        k_new(:,3) = -abs(k_new(:,3));
    end
    idx = randi(length(Epart), injCount, 1);
    norm_factors = params.knorm(Epart(idx)-params.Egap, m_eff);
    k_new = k_new .* norm_factors(:);
    k = [k; k_new];
    if strcmp(side, 'left')
        new_r = params.realspace(1) + params.del_t_eh*(params.hbar*k_new(:,3)/(m_eff*params.m_0));
    else
        new_r = params.device_thickness + params.del_t_eh*(params.hbar*k_new(:,3)/(m_eff*params.m_0));
    end
    r = [r; new_r];
end

function observables = updateObservables(obs, into, injLeft, injRight, particles, params, t)
    obs.charge_collected_ITO(t) = sum(gather(into.hole_into_ITO)) - sum(gather(into.elec_into_ITO));
    obs.charge_collected_Cu(t)  = sum(gather(into.hole_into_Cu)) - sum(gather(into.elec_into_Cu));
    obs.charge_injected_ITO(t)  = injLeft;
    obs.charge_injected_Cu(t)   = injRight;
    obs.J(t) = (obs.charge_collected_Cu(t) - obs.charge_injected_Cu(t) + ...
                obs.charge_injected_ITO(t) - obs.charge_collected_ITO(t)) * params.q / params.del_t_eh;
    obs.Avg_Vel_e(t) = params.hbar * mean(particles.k_e(:,3)) / (params.m_e*params.m_0);
    obs.Avg_Vel_h(t) = params.hbar * mean(particles.k_h(:,3)) / (params.m_h*params.m_0);
    %obs.Psi(:,t) = psi;
    obs.elec_vs_t(:,t) = histcounts(particles.r_e, params.realspace);
    obs.hole_vs_t(:,t) = histcounts(particles.r_h, params.realspace);
    observables = obs;
end

function plotInitialFields(params, Efield, particles)
    figure;
    plot(params.realspace*1e9, Efield);
    title('Initial Electric Field at t=0'); xlabel('Position (nm)'); ylabel('Electric Field (V/m)');
    
    figure;
    plot(params.Ep/params.q - params.Egap/params.q, params.prob_dist_e, ...
         params.Ep/params.q - params.Egap/params.q, params.prob_dist_h);
    xlim([params.Ep(1)/params.q - params.Egap/params.q, params.Emax/params.q - params.Egap/params.q]);
    
    figure;
    energy = linspace(params.Egap, params.Egap+params.Emax, 1000);
    electron_Ekin = arrayfun(@(i) params.Ekin(particles.k_e(i,:), params.m_e), 1:length(particles.k_e))';
    hole_Ekin = arrayfun(@(i) params.Ekin(particles.k_h(i,:), params.m_h), 1:length(particles.k_h))';
    Eedensity = cloudincell(electron_Ekin+params.Egap, energy);
    Ehdensity = cloudincell(hole_Ekin+params.Egap, energy);
    plot(energy/params.q, Eedensity, energy/params.q, Ehdensity);
    xlabel('Kinetic Energy (eV)'); ylabel('Density per unit energy'); 
    title('Carrier Probability Distribution'); legend('Electrons','Holes');
end

function computeQuasiFermiLevels(particles, params)
    Nquasi = 200;
    e_density = cloudincell(particles.r_e, linspace(0, params.device_thickness, Nquasi));
    h_density = cloudincell(particles.r_h, linspace(0, params.device_thickness, Nquasi));
    quasi_Efn = zeros(1, Nquasi); quasi_Efp = zeros(1, Nquasi);
    parfor i = 1:Nquasi
        [qFn, qFp] = quasifermifind(e_density(i), h_density(i), params.m_e, params.m_h, params.Egap);
        quasi_Efn(i) = qFn; quasi_Efp(i) = qFp;
    end
    figure;
    plot(linspace(0,100,Nquasi), quasi_Efn/params.q, linspace(0,100,Nquasi), quasi_Efp/params.q, ...
         linspace(0,100,Nquasi), params.Ef/params.q*ones(1,Nquasi));
    xlabel('Device position (nm)'); ylabel('Energy (eV)'); legend('Electrons','Holes','Intrinsic');
end

function plotPostProcessing(obs, params)
    t_vec = linspace(0, params.Ntime*params.del_t_eh, params.Ntime);
    
    figure;
    plot(t_vec, smooth(smooth(obs.J)), t_vec, obs.J); grid on;
    title('Current vs Time'); xlabel('Time (s)'); ylabel('Current (A/m^2)');
    
    figure;
    plot(t_vec, sum(obs.elec_vs_t,1), t_vec, sum(obs.hole_vs_t,1)); grid on;
    title('Number of Charges vs Time'); xlabel('Time (s)'); ylabel('Carrier particles'); legend({'Electrons','Holes'});
    
    charge_inout = obs.charge_injected_ITO + obs.charge_injected_Cu - obs.charge_collected_Cu - obs.charge_collected_ITO;
    figure;
    plot(t_vec, charge_inout); grid on;
    title('Charges through Contacts vs Time'); xlabel('Time (s)'); ylabel('Carrier particles');
    
    figure;
    mesh(obs.hole_vs_t); view(2); title('Hole Particle Density vs Time'); xlabel('Time step'); ylabel('Position (Angstrom)'); colorbar;
    
    figure;
    mesh(obs.elec_vs_t); view(2); title('Electron Particle Density vs Time'); xlabel('Time step'); ylabel('Position (Angstrom)'); colorbar;
end

function density = cloudincell(dist, xaxis, weight)
    if nargin < 3, weight = ones(size(dist)); end
    dist = dist(:); weight = weight(:); xaxis = xaxis(:);
    dx = diff(xaxis);
    idx = discretize(dist, xaxis);
    idx(isnan(idx)) = length(xaxis)-1;
    frac = (dist - xaxis(idx)) ./ dx(idx);
    density = accumarray(idx, weight .* (1-frac) ./ dx(idx), [length(xaxis),1]) + ...
              accumarray(idx+1, weight .* frac ./ dx(idx), [length(xaxis),1]);
    density = density.';
end

function density = cloudincell_gpu2(dist, xaxis, weight)
    if nargin < 3, weight = ones(size(dist), 'like', dist); end
    dist = gpuArray(dist(:)); xaxis = gpuArray(xaxis(:)); weight = gpuArray(weight(:));
    if isempty(dist)
        error('Input distribution is empty.');
    elseif max(dist) > max(xaxis) || min(dist) < min(xaxis)
        warning('Some distribution values are outside the xaxis range.');
    end
    dx = abs(xaxis(2) - xaxis(1));
    scale = 1/dx^2;
    d = abs(dist - xaxis');
    dw = abs(d) .* weight;
    pos = (d <= dx) & (d > 0);
    density = gather(sum(dw .* pos * scale, 1));
end

function [total_e, total_h] = fermifinder2(m_e, m_h, Na, Eta, qa, Nd, qd, Etd, Egap, T)
    m0 = 9.109534e-31; kb = 1.380662e-23; hbar = 1.0545887e-34; q = 1.6021892e-19; S = 1;
    h = 2*pi*hbar; F = @(x,c) x.^(1/2)./(1+exp(x-c));
    itmax = 10000; tol = Egap/1e18; a = 0; b = Egap; c = (a+b)/2;
    na = 2*(2*pi*m_e*m0*kb*T/(h^2))^(3/2)*integral(@(x)F(x,(a-Egap)/(kb*T)),0,inf);
    pa = 2*(2*pi*m_h*m0*kb*T/(h^2))^(3/2)*integral(@(x)F(x,-a/(kb*T)),0,inf);
    Ndiona = qd.*Nd .* (1 - 1./(1+S*exp((Etd - a)/(kb*T))));
    Naiona = qa.*Na .* (1 - 1./(1+S*exp((Eta - (Egap-a))/(kb*T))));
    diffa = pa - na - sum(Naiona) + sum(Ndiona); itr = 1;
    while itr < itmax
        c = (a+b)/2;
        nc = 2*(2*pi*m_e*m0*kb*T/(h^2))^(3/2)*integral(@(x)F(x,(c-Egap)/(kb*T)),0,inf);
        pc = 2*(2*pi*m_h*m0*kb*T/(h^2))^(3/2)*integral(@(x)F(x,-c/(kb*T)),0,inf);
        Ndionc = qd.*Nd .* (1 - 1./(1+S*exp((Etd - c)/(kb*T))));
        Naionc = qa.*Na ./ (1+S*exp((Eta - c)/(kb*T)));
        diffc = pc - nc - sum(Naionc) + sum(Ndionc);
        if diffc == 0 || (b-a)/2 < tol
            total_e = nc; total_h = pc; disp('Reached convergence'); return
        end
        itr = itr + 1;
        if sign(diffc) == sign(diffa)
            a = c; diffa = diffc;
        else
            b = c;
        end
    end
    disp('Reached iteration limit'); total_e = nc; total_h = pc;
end

function [qFn, qFp] = quasifermifind(e_density, h_density, m_e, m_h, Egap)
    m0 = 9.109534e-31; kb = 1.380662e-23; hbar = 1.0545887e-34; T = 300;
    h = 2*pi*hbar; F = @(x,c) x.^(1/2)./(1+exp(x-c)); itmax = 5000; tol = Egap/1e17; itr = 1;
    ea = 0; eb = Egap;
    na = 2*(2*pi*m_e*m0*kb*T/h^2)^(3/2).*integral(@(x)F(x,(ea-Egap)/kb/T),0,inf);
    ediffa = e_density - na;
    while itr < itmax
        ec = (ea+eb)/2;
        nc = 2*(2*pi*m_e*m0*kb*T/h^2)^(3/2).*integral(@(x)F(x,(ec-Egap)/kb/T),0,inf);
        ediffc = e_density - nc;
        if ediffc == 0 || (eb-ea)/2 < tol, qFn = ec; break, end
        itr = itr + 1;
        if sign(ediffc) == sign(ediffa)
            ea = ec; ediffa = ediffc;
        else
            eb = ec;
        end
    end
    itr = 1; ha = 0; hb = Egap;
    pa = 2*(2*pi*m_h*m0*kb*T/h^2)^(3/2).*integral(@(x)F(x,(0-ha)/kb/T),0,inf);
    hdiffa = h_density - pa;
    while itr < itmax
        hc = (ha+hb)/2;
        pc = 2*(2*pi*m_h*m0*kb*T/h^2)^(3/2).*integral(@(x)F(x,(0-hc)/kb/T),0,inf);
        hdiffc = h_density - pc;
        if hdiffc == 0 || (hb-ha)/2 < tol, qFp = hc; break, end
        itr = itr + 1;
        if sign(hdiffc) == sign(hdiffa)
            ha = hc; hdiffa = hdiffc;
        else
            hb = hc;
        end
    end
    qFn = ec; qFp = hc;
end

function Efield = Efield_solver(psi, dx)
    n = length(psi);
    Efield = zeros(n,1);
    Efield(1) = (-3*psi(1)+4*psi(2)-psi(3))/(2*dx);
    Efield(end) = (psi(end-2)-4*psi(end-1)+3*psi(end))/(2*dx);
    Efield(2) = (psi(3)-psi(1))/(2*dx);
    Efield(end-1) = (psi(end)-psi(end-2))/(2*dx);
    for i = 3:n-2
        Efield(i) = (psi(i-2)-8*psi(i-1)+8*psi(i+1)-psi(i+2))/(12*dx);
    end
end

function V = poisson_solver(M, RHS, Vbias)
    q = 1.6021892e-19; n = length(RHS);
    bx = RHS;
    bx(1) = RHS(1);
    bx(n) = RHS(n) + M(1,1)/2*(Vbias*q);
    V = M\bx';
end

function count = bucket(dist, edges)
    count = sum((dist > edges(1) & dist <= edges(2)) | (dist == 0 & edges(1) == 0));
end

function updateLivePlots(obs, params, t)
    t_vec = linspace(0, t*params.del_t_eh, t);
    
    % Create/update figure with multiple subplots
    figure(101); clf;
    
    % Plot Current vs Time
    subplot(2,2,1);
    plot(t_vec, smooth(obs.J(1:t)), 'LineWidth', 1.5);
    grid on;
    title('Current vs Time');
    xlabel('Time (s)'); ylabel('Current (A/m^2)');
    
    % Plot Carrier Numbers vs Time
    subplot(2,2,2);
    plot(t_vec, sum(obs.elec_vs_t(:,1:t),1), 'b', t_vec, sum(obs.hole_vs_t(:,1:t),1), 'r');
    grid on;
    title('Number of Carriers vs Time');
    xlabel('Time (s)'); ylabel('Carrier Count');
    legend('Electrons','Holes');
    
    % Plot Charge Injection/Collection vs Time
    subplot(2,2,3);
    charge_inout = obs.charge_injected_ITO(1:t) + obs.charge_injected_Cu(1:t) - ...
                   obs.charge_collected_Cu(1:t) - obs.charge_collected_ITO(1:t);
    plot(t_vec, charge_inout, 'LineWidth', 1.5);
    grid on;
    title('Charges through Contacts vs Time');
    xlabel('Time (s)'); ylabel('Charge Count');
    
    % Plot the latest Potential Profile
    subplot(2,2,4);
    plot(params.realspace, obs.Psi(:,t), 'k', 'LineWidth', 1.5);
    grid on;
    title('Potential Profile at Current Step');
    xlabel('Position'); ylabel('Potential (V)');
end
