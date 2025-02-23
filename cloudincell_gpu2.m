function density = cloudincell_gpu2(dist, xaxis, weight)
    % Computes density at positions given by xaxis based on particle positions (dist)
    % and associated weights. All calculations are performed on the GPU.
    %
    % Inputs:
    %   dist   - N×1 vector of particle positions
    %   xaxis  - M×1 vector of positions where density is desired
    %   weight - (optional) N×1 vector of particle weights (default: ones(N,1))
    %
    % Output:
    %   density - 1×M vector of density values (returned to the CPU)
    
    if nargin < 3
        weight = ones(size(dist), 'like', dist);
    end
    
    % Ensure column vectors and move to GPU
    dist   = gpuArray(dist(:));
    xaxis  = gpuArray(xaxis(:));
    weight = gpuArray(weight(:));
    
    if isempty(dist)
        error('Input distribution is empty.');
    elseif max(dist) > max(xaxis) || min(dist) < min(xaxis)
        warning('Some distribution values are outside the xaxis range.');
    end
    
    % Constant spacing assumed for xaxis
    dx = abs(xaxis(2) - xaxis(1));
    scale = 1 / dx^2;
    
    % Compute the difference between each particle and each xaxis point.
    % Resulting d is an N×M matrix.
    d = abs(dist - xaxis');  % implicit expansion
    
    % Weight each difference by the particle weight (implicit expansion)
    dw = abs(d) .* weight;
    
    % Determine which particles are within dx of the xaxis point (but not exactly at 0)
    pos = (d <= dx) & (d > 0);
    
    % Sum contributions along each column, scale appropriately, and gather the result
    density = gather(sum(dw .* pos * scale, 1));
end
