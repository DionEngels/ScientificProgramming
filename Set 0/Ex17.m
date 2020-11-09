help polyval
fprintf('Polynomial p(x) = x^2 - x + 3:\n');
p = [1, -1, 3];
fprintf('Evaluated at x = 0:\n');
polyval(p,0)
fprintf('Determine the characteristic polynomial of a matrix A:\n');
fprintf('Observe what round-off does ...\n');
A = [1 8 -10; -4 2 4; -5 2 8]
poly(A)
fprintf('The Cayley Hamilton theorem states: p(A) = 0 where p is the characteristic polynomial of A:\n');
polyval(p,A) % This goes wrong ... because ...
fprintf('p(x) = x + 3, so polyval() should output p(A) = A + 3*I = A^1 + 3*A^0 = [4,0,0;0,4,0;0,0,4]\n');
fprintf('but it does not. It outputs A + [4,3,3;3,4,3;3,3,4] ...\n');
p = [1, 3], polyval(p,eye(3))