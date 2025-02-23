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