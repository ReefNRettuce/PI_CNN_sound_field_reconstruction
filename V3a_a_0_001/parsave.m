% parsave.m
function parsave(filename, u_grid, X, Y, source_x, source_y, freq, k)
    save(filename, 'u_grid', 'X', 'Y', 'source_x', 'source_y', 'freq', 'k');
end