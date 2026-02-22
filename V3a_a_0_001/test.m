% tests function 

[u_grid, X, Y] = Helmholtz_Solver(-0.5, 1.5, 300, 343, 3.0, 3.0, 32);

%figure plot 

figure
pcolor(X, Y, real(u_grid))
shading interp;
colorbar;
title("Pressure Field")

fprintf('Max pressure: %e\n', max(abs(u_grid(:))));