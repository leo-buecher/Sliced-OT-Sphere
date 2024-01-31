function CRf = cdt(Rf, xp, w, x_cdt, interval)
  if nargin < 3
    w = ones(size(Rf,1), 1);
  end
  if nargin < 4
    x_cdt = xp;
  end

  Rf_reg = Rf;
  if nargin > 4 && abs(interval(1) - xp(1)) + abs(interval(2) - xp(end)) > eps
    if abs(interval(1) - xp(1)) > abs(interval(2) - xp(1))
      warning('interval boundaries might be in the wrong order')
    end
    % Regularize cdf to avoid interpolation problems at the boundary 0,1
    Rf_reg = [zeros(1,size(Rf,2)); Rf; zeros(1,size(Rf,2))];
    w = [0; w; 0];
    xp = [interval(1); xp; interval(2)];
  end

  Rfw = Rf_reg .* w + 1e-8;
  Rfw = Rfw ./ sum(Rfw,1);
  R0 = ones(size(Rf_reg)); % Template
  R0w = R0 .* w + 1e-8;
  R0w = R0w ./ sum(R0w,1);

  cdf0 = cumtrapz(R0w,1);
  cdf1 = cumtrapz(Rfw,1);

  x01 = linspace(0,1, 2*numel(x_cdt)).';
  CRf = zeros(length(x_cdt), size(Rf,2));
  for i = 1:size(Rf,2)
    cdf0_inverse = interp1(cdf0(:,i), xp, x01, 'pchip', 'extrap');
    cdf1_inverse = interp1(cdf1(:,i), xp, x01, 'pchip', 'extrap');
    u = interp1(cdf0_inverse, cdf0_inverse-cdf1_inverse, x_cdt, 'pchip', 'extrap');

    CRf(:,i) = u; % .* sqrt(R0(:,1)); % Weight not necessary since it cancels out with inverse
  end
end

