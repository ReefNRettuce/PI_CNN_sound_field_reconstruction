% helmholtz_room_solver.m
% Solves: ∇²u + k²u = 0 with a point source

function[u_grid, X, Y] = Helmholtz_Solver(source_x, source_y, freq, c, room_length, room_width, grid_size)

k = 2*pi*freq / c;  
%% Create geometry (rectangular room)
model = createpde();
R1 = [3; 4; 0; room_length; room_length; 0; 0; 0; room_width; room_width];
g = decsg(R1);
geometryFromEdges(model, g);

%% Boundary conditions
% Absorbing boundary (impedance BC): ∂u/∂n + ik*u = 0
% For simplicity, using mixed BC approximation
applyBoundaryCondition(model, 'neumann', 'Edge', 1:4, 'g', 0, 'q', 1i*k);

%% PDE coefficients
% Helmholtz: -∇²u - k²u = f  →  in MATLAB form: -∇·(c∇u) + au = f
% So c = 1, a = -k², f = point source
specifyCoefficients(model, 'm', 0, 'd', 0, 'c', 1, 'a', -k^2, 'f', @(location,state) sourceFunction(location, source_x, source_y));

%% Generate mesh
generateMesh(model, 'Hmax', 0.05);  % fine mesh

%% Solve
result = solvepde(model);

%% Interpolate to regular grid
x = linspace(0, room_length, grid_size);
y = linspace(0, room_width, grid_size);
[X, Y] = meshgrid(x, y);
u = interpolateSolution(result, X(:), Y(:));
u_grid = reshape(u, grid_size, grid_size);

% Normalize to max amplitude of 1
max_amp = max(abs(u_grid(:)));
if max_amp > 0
    u_grid = u_grid / max_amp;
end



