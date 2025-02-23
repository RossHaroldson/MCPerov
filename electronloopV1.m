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