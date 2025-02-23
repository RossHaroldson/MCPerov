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