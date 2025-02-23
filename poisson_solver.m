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