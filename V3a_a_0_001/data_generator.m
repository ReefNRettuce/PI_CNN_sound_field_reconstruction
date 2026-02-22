% Parameters
freq = 300;
c = 343;
k = 2*pi*freq / c;
room_length = 3.0;
room_width = 3.0;
grid_size = 32;

% Multiple distances from room edge
distances = [-0.5, -1.0, -1.5, -2.0];
points_per_side = 256;

% Initialize arrays
all_x = [];
all_y = [];

for d = distances
    % Left side: x = d, y varies
    all_x = [all_x, repmat(d, 1, points_per_side)];
    all_y = [all_y, linspace(0, 3, points_per_side)];
    
    % Right side: x = 3 - d, y varies
    all_x = [all_x, repmat(3 - d, 1, points_per_side)];
    all_y = [all_y, linspace(0, 3, points_per_side)];
    
    % Bottom: y = d, x varies
    all_x = [all_x, linspace(0, 3, points_per_side)];
    all_y = [all_y, repmat(d, 1, points_per_side)];
    
    % Top: y = 3 - d, x varies
    all_x = [all_x, linspace(0, 3, points_per_side)];
    all_y = [all_y, repmat(3 - d, 1, points_per_side)];
end

% Total: 4 distances × 4 sides × 32 points = 512 samples
% To get closer to 1000, increase points_per_side to 63 (gives 1008)


filepath = '/Users/leifefrancisco/Documents/Workspace/PI_CNN_V2/PI_CNN_V3a/V3a_a_0_001/data/room_acoustic_data_room_%04d.mat';


num_samples = length(all_x);
fprintf('Generating %d samples\n', num_samples);

parpool('Processes');

parfor i = 1:num_samples
    source_x = all_x(i);
    source_y = all_y(i);
    
    [u_grid, X, Y] = Helmholtz_Solver(source_x, source_y, freq, c, room_length, room_width, grid_size);
    
    filename = sprintf(filepath, i);
    parsave(filename, u_grid, X, Y, source_x, source_y, freq, k);
end

disp('Done!');