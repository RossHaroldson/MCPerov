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
psi = V/params.q;
Efield = Efield_solver(psi, params.dx);
disp('Initial potential and field solved');

plotInitialFields(params, Efield, particles);

%% Preallocate Observables
observables = initObservables(params, length(params.realspace));

tic;
for t = 1:params.Ntime
    % Check dt criteria
    checkDt(particles, params);
    
    % Update potential and field
    rho = (cloudincell_gpu2(particles.r_h, params.realspace) - cloudincell_gpu2(particles.r_e, params.realspace)) * params.q;
    V = poisson_solver(poismatrix, params.q/params.eps * rho, params.Vbias);
    psi = V/params.q;
    Efield = Efield_solver(psi, params.dx);
    
    % Compute scattering rates and GPU arrays
    [particles, scatRates, gpu_Efield_e, gpu_Efield_h, gpu_rand_e, gpu_rand_h] = updateScattering(particles, params, Efield);
    
    % Update particle positions using GPU loops
    [particles.r_e, particles.k_e, into.elec_into_Cu, into.elec_into_ITO] = updateParticles_gpu('electron', particles.r_e, particles.k_e, params, gpu_Efield_e, gpu_rand_e, scatRates.e);
    [particles.r_h, particles.k_h, into.hole_into_Cu, into.hole_into_ITO] = updateParticles_gpu('hole', particles.r_h, particles.k_h, params, gpu_Efield_h, gpu_rand_h, scatRates.h);
    
    % Remove particles that went out-of-bound (NaN entries)
    [particles.r_e, particles.k_e] = removeNaNParticles(particles.r_e, particles.k_e);
    [particles.r_h, particles.k_h] = removeNaNParticles(particles.r_h, particles.k_h);
    
    % Particle injection at left/right boundaries
    [particles.r_e, particles.k_e, injLeft, injRight] = injectParticles(particles.r_e, particles.k_e, params, energyStruct.Epart_e, 'electron');
    [particles.r_h, particles.k_h, injLeft, injRight] = injectParticles(particles.r_h, particles.k_h, params, energyStruct.Epart_h, 'hole');
    
    % Update observables (current, velocity, density histograms, etc.)
    observables = updateObservables(observables, V, into, injLeft, injRight, particles, params, t);
    
    disp(t);
end
fprintf('Time steps per second: %.2f\n', params.Ntime/toc);

%% Compute Quasi-Fermi Levels & Post-Processing Plots
computeQuasiFermiLevels(particles, params);
plotPostProcessing(observables, params);

%% --- Local Helper Functions ---
function params = initParameters()
    % Define physical constants
    params.im       = sqrt(-1);
    params.eps_0    = 8.854187818e-12;   % Vacuum permittivity (F/m)
    params.m_0      = 9.109534e-31;      % Electron rest mass (kg)
    params.hbar     = 1.0545887e-34;     % Reduced Planck constant (J*s)
    params.h        = 2*pi*params.hbar;
    params.kb       = 1.380662e-23;      % Boltzmann constant (J/K)
    params.q        = 1.6021892e-19;     % Electron charge (C)
    params.a0       = 5.2917706e-11;     % Bohr radius (m)
    
    % Define simulation parameters
    params.T           = 300;  
    params.a           = 6.017e-10;
    params.N_0         = 1/params.a^3;
    params.density     = 4420;
    params.Ntime       = 3000;
    params.N_e         = 10000;
    params.m_e         = 0.149;
    params.N_h         = 10000;
    params.m_h         = 0.143;
    params.Econd       = -2.93;
    params.Eval        = -5.3;
    params.DFE_Ai     = 1.03*params.q;
    params.N_Ai       = 100;
    params.E_Ai       = (-5.2 - params.Eval)*params.q;
    params.DFE_Va     = 1.1*params.q;
    params.N_Va       = 100;
    params.E_Va       = (-3.03 - params.Eval)*params.q;
    params.nonpar_e   = 1.033;
    params.nonpar_h   = 1.054;
    params.Egap       = (params.Econd - params.Eval)*params.q;
    params.device_thickness = 1000e-10;
    params.Ef         = params.Egap/2 + 3/4*params.kb*params.T*log(params.m_h/params.m_e);
    
    % Calculate densities using fermifinder2 (assumed available)
    total_Ai_density = params.N_0*exp(-params.DFE_Ai/(params.kb*params.T));
    total_Va_density = params.N_0*exp(-params.DFE_Va/(params.kb*params.T));
    [total_e_density, total_h_density, ~, ~, ~] = fermifinder2(params.m_e, params.m_h, total_Ai_density, params.E_Ai, 1, total_Va_density, 1, params.E_Va, params.Egap, params.T);
    params.stat_weight_e = params.device_thickness*total_e_density/params.N_e;
    params.stat_weight_h = params.device_thickness*total_h_density/params.N_h;
    
    params.eps_rel_inf   = 4.3;
    params.w_TO_1       = 4.83e-3*params.q;
    params.w_TO_2       = 11.66e-3*params.q;
    params.w_LO_1       = 5.62e-3*params.q;
    params.w_LO_2       = 19.42e-3*params.q;
    params.d_e          = -2.93*params.q;
    params.d_h          = -2.2*params.q;
    params.C            = 2.1e10;
    params.C_MAPbBr3_11 = 3.13e10;
    params.C_MAPbBr3_44 = 4.3e9;
    params.eps          = params.eps_rel_inf*params.eps_0;
    
    % Spatial grid and bias
    params.npoints      = 1000;
    params.realspace    = linspace(0, params.device_thickness, params.npoints);
    params.dx           = params.realspace(2) - params.realspace(1);
    params.ITO_workfunction = -4.7;
    params.InGa_workfunction = -4.2;
    params.Vbias        = 1;
    
    % Time steps and scattering
    params.del_t_eh    = 5e-17;
    params.del_t_ions  = 1e-15;
    params.kmaxnorm    = params.dx/params.del_t_eh * params.m_e*params.m_0/params.hbar;
    params.Emax        = 0.42e-18;
    params.Ekin        = @(k, m) params.hbar^2 * sum(k.^2, 2) / (2*m*params.m_0);
    params.gradE       = @(k, m) params.hbar^2 * sqrt(sum(k.^2, 2))/(m*params.m_0);
    params.knorm       = @(E, m) sqrt(2*m*params.m_0*E)/params.hbar;
    params.v_k         = @(k, m) params.gradE(k, m)/params.hbar;
    
    % Acoustic scattering functions
    params.P_e_ac = @(Ekin) params.d_e^2 * sqrt(2)*(params.m_e*params.m_0)^(3/2)*params.kb*params.T*sqrt(Ekin) / (pi*params.hbar^4*params.C);
    params.P_h_ac = @(Ekin) params.d_h^2 * sqrt(2)*(params.m_h*params.m_0)^(3/2)*params.kb*params.T*sqrt(Ekin) / (pi*params.hbar^4*params.C);
    params.ac_scat_scale = 1e2;
    
    % Energy distributions for electrons and holes
    Ep = exp(linspace(log(params.Egap), log(params.Egap+params.Emax), 100000));
    Ep(1) = params.Egap;
    density_factor = 8*pi*sqrt(2)/(params.hbar*2*pi)^3;
    params.prob_dist_e = density_factor*(params.m_e*params.m_0)^(3/2)*sqrt(Ep-params.Egap)./(1+exp((Ep-params.Ef)/(params.kb*params.T)));
    params.prob_dist_h = density_factor*(params.m_h*params.m_0)^(3/2)*sqrt(Ep-params.Egap)./(1+exp((Ep-params.Ef)/(params.kb*params.T)));
    params.Ep = Ep;
end

function [particles, energyStruct] = initParticles(params)
    % Initialize electron and hole positions and normalized k-vectors
    particles.r_e = rand(params.N_e, 1) * params.device_thickness;
    particles.k_e = normalizeRows(rand(params.N_e, 3)*2 - 1);
    particles.r_h = rand(params.N_h, 1) * params.device_thickness;
    particles.k_h = normalizeRows(rand(params.N_h, 3)*2 - 1);
    
    % Energy partition sampling using provided distributions
    energyStruct.Epart_e = randpdf(params.prob_dist_e, params.Ep, [1, params.N_e]);
    energyStruct.Epart_h = randpdf(params.prob_dist_h, params.Ep, [1, params.N_h]);
    % Set the magnitude of each k-vector based on the sampled energy
    for i = 1:params.N_e
        particles.k_e(i,:) = particles.k_e(i,:) * params.knorm(energyStruct.Epart_e(i)-params.Egap, params.m_e);
    end
    for i = 1:params.N_h
        particles.k_h(i,:) = particles.k_h(i,:) * params.knorm(energyStruct.Epart_h(i)-params.Egap, params.m_h);
    end
end

function M = normalizeRows(M)
    norms = sqrt(sum(M.^2, 2));
    M = M ./ norms;
end

function [poismatrix, alpha] = buildPoissonMatrix(dx, n)
    alpha = 1/dx^2;
    poismatrix = 2*alpha*eye(n) - alpha*diag(ones(n-1,1),1) - alpha*diag(ones(n-1,1),-1);
end

function observables = initObservables(params, n)
    observables.J                     = zeros(1, params.Ntime);
    observables.charge_collected_ITO  = zeros(1, params.Ntime);
    observables.charge_collected_Cu   = zeros(1, params.Ntime);
    observables.Avg_Vel_e             = zeros(1, params.Ntime);
    observables.Avg_Vel_h             = zeros(1, params.Ntime);
    observables.Psi                   = zeros(n, params.Ntime);
    observables.elec_vs_t             = zeros(n-1, params.Ntime);
    observables.hole_vs_t             = zeros(n-1, params.Ntime);
    observables.charge_injected_ITO   = zeros(1, params.Ntime);
    observables.charge_injected_Cu    = zeros(1, params.Ntime);
    observables.elec_into_Cu          = zeros(1,params.Ntime);
    observables.hole_into_Cu          = zeros(1,params.Ntime);
    observables.elec_into_ITO         = zeros(1,params.Ntime);
    observables.hole_into_ITO         = zeros(1,params.Ntime);
    observables.injLeft               = zeros(1,params.Ntime);
    observables.injRight              = zeros(1,params.Ntime);
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
    gpu_rand_e   = rand(length(particles.r_e), 1, 'gpuArray');
    gpu_rand_h   = rand(length(particles.r_h), 1, 'gpuArray');
end

function [r_new, k_new, into_Cu, into_ITO] = updateParticles_gpu(type, r, k, params, gpu_Efield, gpu_rand, scatter)
    if strcmp(type, 'electron')
        [r_new, kx_new, ky_new, kz_new, into_Cu, into_ITO] = ...
            arrayfun(@electronloopV1, gpuArray(r), gpuArray(k(:,1)), gpuArray(k(:,2)), gpuArray(k(:,3)), ...
                     params.m_e, params.del_t_eh, gpuArray(scatter), params.device_thickness, gpu_Efield, gpu_rand);
    else
        [r_new, kx_new, ky_new, kz_new, into_Cu, into_ITO] = ...
            arrayfun(@holeloopV1, gpuArray(r), gpuArray(k(:,1)), gpuArray(k(:,2)), gpuArray(k(:,3)), ...
                     params.m_h, params.del_t_eh, gpuArray(scatter), params.device_thickness, gpu_Efield, gpu_rand);
    end
    r_new = gather(r_new);
    k_new = [gather(kx_new) gather(ky_new) gather(kz_new)];
end

function [r_clean, k_clean] = removeNaNParticles(r, k)
    valid = ~isnan(r);
    r_clean = r(valid);
    k_clean = k(valid, :);
end

function [r, k, injLeft, injRight] = injectParticles(r, k, params, Epart, type)
    % Determine bucket counts at the left/right boundaries
    leftCount  = bucket(r, [params.realspace(1) params.realspace(2)]);
    rightCount = bucket(r, [params.realspace(end-1) params.realspace(end)]);
    % For simplicity we assume desired count is proportional to total particle number
    if strcmp(type, 'electron')
        desired = params.N_e/params.npoints;
    else
        desired = params.N_h/params.npoints;
    end
    injLeft  = max(0, desired - leftCount);
    injRight = max(0, desired - rightCount);
    if injLeft > 0
        [r, k] = injectBoundary(r, k, injLeft, Epart, params, type, 'left');
    end
    if injRight > 0
        [r, k] = injectBoundary(r, k, injRight, Epart, params, type, 'right');
    end
end

function [r, k] = injectBoundary(r, k, injCount, Epart, params, type, side)
    k_dum = rand(injCount, 3)*2 - 1;
    if strcmp(side, 'right')
        k_dum(:,3) = -abs(k_dum(:,3));
    end
    idx = randi(length(Epart), injCount, 1);
    for i = 1:injCount
        % Choose effective mass based on carrier type
        m_eff = strcmp(type, 'electron')*params.m_e + strcmp(type, 'hole')*params.m_h;
        k_dum(i,:) = k_dum(i,:) * params.knorm(Epart(idx(i))-params.Egap, m_eff);
    end
    k = [k; k_dum];
    if strcmp(side, 'left')
        new_r = params.realspace(1) + params.del_t_eh * (params.hbar * k_dum(:,3) / (m_eff*params.m_0));
    else
        new_r = params.device_thickness + params.del_t_eh * (params.hbar * k_dum(:,3) / (m_eff*params.m_0));
    end
    r = [r; new_r];
end

function observables = updateObservables(observables, V, into, injLeft, injRight, particles, params, t)
    % Update contact-related currents and average velocities
    % (Assumes that the GPU-returned injection/collection data is incorporated here)
    observables.charge_collected_ITO(t) = sum(gather(into.hole_into_ITO)) - sum(gather(into.elec_into_ITO));  % (update with your computed values)
    observables.charge_collected_Cu(t)  = sum(gather(into.hole_into_Cu)) - sum(gather(into.elec_into_Cu));
    observables.charge_injected_ITO(t)  = injLeft;
    observables.charge_injected_Cu(t)   = injRight;
    observables.J(t) = (observables.charge_collected_Cu(t) - observables.charge_injected_Cu(t) + ...
                        observables.charge_injected_ITO(t) - observables.charge_collected_ITO(t)) * params.q / params.del_t_eh;
    observables.Avg_Vel_e(t) = params.hbar * mean(particles.k_e(:,3)) / (params.m_e*params.m_0);
    observables.Avg_Vel_h(t) = params.hbar * mean(particles.k_h(:,3)) / (params.m_h*params.m_0);
    observables.Psi(:,t) = V./params.q;  % (if you wish to store the potential)
    observables.elec_vs_t(:,t) = histcounts(particles.r_e, params.realspace);
    observables.hole_vs_t(:,t) = histcounts(particles.r_h, params.realspace);
end

function plotInitialFields(params, Efield, particles)
    figure;
    plot(params.realspace*1e9, Efield);
    title('Initial Electric Field at t=0');
    xlabel('Position (nm)'); ylabel('Electric Field (V/m)');
    
    figure;
    plot(params.Ep/params.q - params.Egap/params.q, params.prob_dist_e, ...
         params.Ep/params.q - params.Egap/params.q, params.prob_dist_h);
    xlim([params.Ep(1)/params.q - params.Egap/params.q, params.Emax/params.q - params.Egap/params.q]);
    
    figure;
    energy = linspace(params.Egap, params.Egap+params.Emax, 1000);
    electron_Ekin = arrayfun(@(i) params.Ekin(particles.k_e(i,:), params.m_e), 1:length(particles.k_e))';
    hole_Ekin     = arrayfun(@(i) params.Ekin(particles.k_h(i,:), params.m_h), 1:length(particles.k_h))';
    Eedensity     = cloudincell(electron_Ekin+params.Egap, energy);
    Ehdensity     = cloudincell(hole_Ekin+params.Egap, energy);
    plot(energy/params.q, Eedensity, energy/params.q, Ehdensity);
    xlabel('Kinetic Energy (eV)'); ylabel('Density per unit energy');
    title('Carrier Probability Distribution');
    legend('Electrons','Holes');
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
    xlabel('Device position (nm)'); ylabel('Energy (eV)');
    legend('Electrons','Holes','Intrinsic');
end

function plotPostProcessing(observables, params)
    t_vec = linspace(0, params.Ntime*params.del_t_eh, params.Ntime);
    
    figure;
    plot(t_vec, smooth(smooth(observables.J)), t_vec, observables.J);
    grid on;
    title('Current vs Time');
    xlabel('Time (s)'); ylabel('Current (A/m^2)');
    
    figure;
    plot(t_vec, sum(observables.elec_vs_t,1), t_vec, sum(observables.hole_vs_t,1));
    grid on;
    title('Number of Charges vs Time');
    xlabel('Time (s)'); ylabel('Carrier particles');
    legend({'Electrons','Holes'});
    
    charge_inout = observables.charge_injected_ITO + observables.charge_injected_Cu - ...
                   observables.charge_collected_Cu - observables.charge_collected_ITO;
    figure;
    plot(t_vec, charge_inout);
    grid on;
    title('Number of Charges passing through contacts vs Time');
    xlabel('Time (s)'); ylabel('Carrier particles');
    
    figure;
    mesh(observables.hole_vs_t); view(2);
    title('Hole Particle Density vs Time');
    xlabel('Time step'); ylabel('Position in device (Angstrom)'); colorbar;
    
    figure;
    mesh(observables.elec_vs_t); view(2);
    title('Electron Particle Density vs Time');
    xlabel('Time step'); ylabel('Position in device (Angstrom)'); colorbar;
end

function density = cloudincell(dist, xaxis, weight)
    % This function computes the density at each point in xaxis
    if nargin < 3
        weight = ones(size(dist));
    end
    if isempty(dist)
        error('Input distribution is empty');
    end
    
    % Ensure inputs are column vectors
    dist   = dist(:);
    weight = weight(:);
    xaxis  = xaxis(:);
    
    dx = diff(xaxis);  % dx is now a column vector
    
    % Determine bin indices for each particle
    idx = discretize(dist, xaxis);
    % For particles exactly at the upper edge, assign them to the previous bin
    idx(isnan(idx)) = length(xaxis)-1;
    
    frac = (dist - xaxis(idx)) ./ dx(idx);
    
    % Distribute weight between the lower and upper cell using accumarray
    density = accumarray(idx, weight .* (1-frac) ./ dx(idx), [length(xaxis),1]) + ...
              accumarray(idx+1, weight .* frac ./ dx(idx), [length(xaxis),1]);
    density = density.';
end


function density = cloudincell_gpu2(dist, xaxis, weight)
    % Computes density at positions given by xaxis based on particle positions (dist)
    % and associated weights. All calculations are performed on the GPU.
    %
    % Inputs:
    %   dist   - N×1 vector of particle positions
    %   xaxis  - M×1 vector of positions where density is desired
    %   weight - (optional) N×1 vector of particle weights (default: ones(N,1))
    %
    % Output:
    %   density - 1×M vector of density values (returned to the CPU)
    
    if nargin < 3
        weight = ones(size(dist), 'like', dist);
    end
    
    % Ensure column vectors and move to GPU
    dist   = gpuArray(dist(:));
    xaxis  = gpuArray(xaxis(:));
    weight = gpuArray(weight(:));
    
    if isempty(dist)
        error('Input distribution is empty.');
    elseif max(dist) > max(xaxis) || min(dist) < min(xaxis)
        warning('Some distribution values are outside the xaxis range.');
    end
    
    % Constant spacing assumed for xaxis
    dx = abs(xaxis(2) - xaxis(1));
    scale = 1 / dx^2;
    
    % Compute the difference between each particle and each xaxis point.
    % Resulting d is an N×M matrix.
    d = abs(dist - xaxis');  % implicit expansion
    
    % Weight each difference by the particle weight (implicit expansion)
    dw = abs(d) .* weight;
    
    % Determine which particles are within dx of the xaxis point (but not exactly at 0)
    pos = (d <= dx) & (d > 0);
    
    % Sum contributions along each column, scale appropriately, and gather the result
    density = gather(sum(dw .* pos * scale, 1));
end

function [r_e, xe, ye, ze, elec_into_Cu, elec_into_ITO] = electronloopV1(r_e,xe,ye,ze,m_e,del_t_eh,tau_e,device_thickness,Efield_int,r)
m_0   = 9.109534e-31;    % free electron mass (kg)
hbar = 1.0545887e-34;   % Planck constant/(2pi) (J*s)
q    = 1.6021892e-19;   % Proton charge (C)
elec_into_Cu = 0;elec_into_ITO = 0;
    %%%%% check for interaction or collision
    if del_t_eh*tau_e>r % collision (interation based on k_vector)
        % this should actually be del_t_eh/tau > rand
        %choose scattering process
        % give new r and k values;
        % Longitudinal Scatter
        % make random k unit vector
        vecnorm = sqrt(xe^2 +ye^2+ze^2);
        xe_dum = 2*rand-1; ye_dum = 2*rand-1; ze_dum = 2*rand-1;
        dumnorm = sqrt(xe_dum^2 +ye_dum^2 + ze_dum^2);
        xe=xe_dum/dumnorm*vecnorm; ye=ye_dum/dumnorm*vecnorm; ze=ze_dum/dumnorm*vecnorm;
        ze = ze - q/hbar .*Efield_int*del_t_eh; %update new k vector
        r_e = r_e + hbar*ze/m_e/m_0*del_t_eh - q/hbar .*Efield_int.*del_t_eh.^2; % updates new position
    elseif 0==1 % interaction based of position
        % recombination of electrons with holes that have the same k_vector
        % trap filling
    else % free flight
        ze = ze - q/hbar .*Efield_int*del_t_eh; %update new k vector
        r_e= r_e + hbar*ze/m_e/m_0*del_t_eh - q/hbar .*Efield_int.*del_t_eh.^2; % updates new position along the z axis
    end
    %%%%% collect electrons if they travel past the device
    if r_e > device_thickness % if electron is in InGa
        r_e=NaN;xe=NaN;ye=NaN;ze=NaN;
        elec_into_Cu = 1;
    elseif r_e < 0 % if electron is in ITO
        r_e=NaN;xe=NaN;ye=NaN;ze=NaN;
        elec_into_ITO = 1;
    end
end

function [r_h, xh, yh, zh, hole_into_Cu, hole_into_ITO] = holeloopV1(r_h,xh,yh,zh,m_h,del_t_eh,tau_h,device_thickness,Efield_int,r)
m_0   = 9.109534e-31;    % free electron mass (kg)
hbar = 1.0545887e-34;   % Planck constant/(2pi) (J*s)
q    = 1.6021892e-19;   % Proton charge (C)
hole_into_Cu = 0;hole_into_ITO = 0;
    %%%%% check for interaction or collision
    if del_t_eh*tau_h>r % collision (interation based on k_vector)
        % this should actually be del_t_eh/tau > rand
        %choose scattering process
        % give new r and k values;
        % Longitudinal Scatter
        % make random k unit vector
        vecnorm = sqrt(xh^2 +yh^2+zh^2);
        xh_dum = 2*rand-1; yh_dum = 2*rand-1; zh_dum = 2*rand-1;
        dumnorm = sqrt(xh_dum^2 +yh_dum^2 + zh_dum^2);
        xh=xh_dum/dumnorm*vecnorm; yh=yh_dum/dumnorm*vecnorm; zh=zh_dum/dumnorm*vecnorm;
        zh = zh + q/hbar .*Efield_int*del_t_eh; %update new k vector
        r_h = r_h + hbar*zh/m_h/m_0*del_t_eh + q/hbar .*Efield_int.*del_t_eh.^2; % updates new position
    elseif 0==1 % interaction based of position
        % recombination of electrons with holes that have the same k_vector
        % trap filling
    else % free flight
        zh = zh + q/hbar .*Efield_int*del_t_eh; %update new k vector
        r_h= r_h + hbar*zh/m_h/m_0*del_t_eh + q/hbar .*Efield_int.*del_t_eh.^2; % updates new position along the z axis
    end
    %%%%% collect electrons if they travel past the device
    if r_h > device_thickness % if electron is in InGa
        r_h=NaN;xh=NaN;yh=NaN;zh=NaN;
        hole_into_Cu = 1;
    elseif r_h < 0 % if electron is in ITO
        r_h=NaN;xh=NaN;yh=NaN;zh=NaN;
        hole_into_ITO = 1;
    end
end

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

function Efield = Efield_solver(psi,dx)
% this function assumes that the length of Efield is 6 or more
Efield=zeros(length(psi),1);
% three point forward difference method
Efield(1) = (-3*psi(1)+4*psi(2)-psi(3))/(2*dx);
% three point backward difference method
Efield(end) = (psi(end-2)-4*psi(end-1)+3*psi(end))/(2*dx);
% two point central difference method
Efield(2) = (psi(3)-psi(1))/(2*dx);
% two point central difference method
Efield(end - 1) = (psi(end)-psi(end-2))/(2*dx);

for i=3:length(psi)-2
        % do 4 point central difference method
        Efield(i)=(psi(i-2)-8*psi(i-1)+8*psi(i+1)-psi(i+2))/(12*dx);
end
end

function V = poisson_solver(poismatrix,RHS,Vbias)
q	= 1.6021892e-19;   % Proton charge (C)
% make rhs of lin system
n=length(RHS);
bx = RHS;
%     ... and the boundary conditions
bx(1) = RHS(1);
bx(n) = RHS(n) + poismatrix(1,1)/2*(Vbias*q);
%V = linsolve(poismatrix,bx'); % older slower method
V = poismatrix\bx';
end

function bin = bucket(dist,edges)
bin=0;
for i = 1:length(dist)
    if (dist(i) - edges(1)) > 0 && (dist(i) - edges(2)) <= 0
        bin = bin + 1;
    end
    if dist(i) == 0 && edges(1)==0
        bin = bin + 1;
    end
end
end