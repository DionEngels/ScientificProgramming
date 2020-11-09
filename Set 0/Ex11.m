help sort
help unique
help uniquetol
v = [3, 1, 4, 1, 2, 3]
sort(v)
unique(v)
unique(v, 'stable')
min(v)
max(v)
fprintf('Determine the different (unique) entries in v = [1, 1+10^-8]:\n');
v = [1, 1+10^-8]
unique(v)
fprintf('Determine the different (unique) entries in v, but assume v_i == v_j iff abs(v_i-v_j)<tol*max(v):\n');
uniquetol(v, 10^-6)

v = [3, 1, 4, 1, 2, 3]
[w, I] = sort(v)
v(I)
[w, I] = unique(v)
v(I)
[w, I] = unique(v, 'stable')
v(I)
[w, I] = min(v)
v(I)
x = 5, y = 6
v = [1, 4, 5, 7, 8, 10, 2]
w = [3, 6, 7, 9, 1, 4]
fprintf('v cup w/v union w: all x in v or in w\n');
union(v, w)
unique([v,w])

T = [];
fprintf('Determine smallest entry in v = randi([1,9],[1,10^7]):\n');
v = randi([1,9],[1,10^7]);
tic; mn = min(v), toc;
fprintf('Determine smallest entry first location in v = randi([1,9],[1,10^7]) -- with unique:\n');
tic; [mn, p] = unique(v); sP = size(p), p1 = p(1), T = [T, toc];
fprintf('Determine smallest entry first location in v = randi([1,9],[1,10^7]) -- with sort:\n');
tic; [mn, p] = sort(v); sP = size(p), p1 = p(1), T = [T, toc];
fprintf('Determine smallest entry first location in v = randi([1,9],[1,10^7]) -- with accumarray:\n');
tic; p = accumarray(v',(1:10^7)',[9,1],@min), p = p(find(p,1)); T = [T, toc]; % watch out, here only 1 index of minimal
fprintf('Determine smallest entry first location in v = randi([1,9],[1,10^7]) -- with min:\n');
tic; [mn, p] = min(v); sP = size(p), p1 = p(1), T = [T, toc]; % watch out, here only 1 index of minimal value
fprintf('Determine ALL smallest entries locations in v = randi([1,9],[1,10^7]) -- with find and min:\n');
tic; R = v; mn = min(R), l = find(R == mn); T = [T, toc];
fprintf('Determine ALL smallest entries locations in v = randi([1,9],[1,10^7]) -- with find and min, store logical position')
tic; R = v; mn = min(R), l = (R == mn); l = find(l); T = [T, toc];
h = bar(diag(T),length(T));
title('Time to determine the location(s) of the smallest integer(s) in a vector of size 10^7')
ylabel('seconds')
grid on
legend('unique','sort','accumarray','min','find/min','find/min+save')
pause