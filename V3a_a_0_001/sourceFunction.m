%% Source function (Gaussian approximation of point source)
function f = sourceFunction(location, sx, sy)
    sigma = 0.05;  % source width
    amplitude = 1000;
    dist_sq = (location.x - sx).^2 + (location.y - sy).^2;
    f = amplitude * exp(-dist_sq / (2*sigma^2));
end