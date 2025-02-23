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
