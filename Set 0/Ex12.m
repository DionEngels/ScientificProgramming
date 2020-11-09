v = randi([1,9],[1,10^1]);
mn = min(v);
minima = v(v==mn);

%% c
% (1:9)(accumarray(v, 2) > 0)
% [mn, index ] = min(v);

%% d
v = randi([1,9],[1,10^3]);
[~, sol, ~] = unique(v);
sol