function Rg = icdt(u, xp, w, x_cdt)
  if nargin < 3
    w = ones(size(xp));
  end
  if nargin < 4
    x_cdt = xp;
  end
  R0 = ones(size(u)); % Template
  R0 = R0 ./ sum(R0,1);
  Rg = zeros(length(xp), size(u,2));
%   u = u ./ sqrt(R0); % Weight not necessary since it cancels out with inverse
  for i = 1:size(u,2)
    f = x_cdt - u(:,i);
    [f0, ia,ic] = unique(f,'last');
    fprime = gradient(f0) ./ gradient(x_cdt(ia));
    I = interp1(f0, R0(ia,i)./fprime, xp, 'pchip', 'extrap');
    I = I/sum(I .* w);
    Rg(:,i) = I;
  end
end