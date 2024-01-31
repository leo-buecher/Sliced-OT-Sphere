% Optimal Transport between probability densities on the 1-dimensional torus of length 1
function [cost, bary, tc] = OTcircle(X, MX,MY, options)
  arguments
    X
    MX
    MY
    options.BaryWeight (1,1) {mustBeNumeric} = 0.5;
    options.TC (1,1) = -1
  end
  X = X(:);
  
  % Slightly regularize to ovoid zero derivative
  MX = MX(:) + 2^40*eps;
  MY = MY(:) + 2^40*eps;
  MX				= MX/sum(MX) ;
  MY				= MY/sum(MY) ;
  
  lambda				= 1 ;
  if options.TC == -1
    [cost,tc] = trans_opt(lambda, X.', X.', MX.', MY.');
  else
    tc = options.TC;
  end
  
  cdf0 = cumsum(MX);
  cdf1 = cumsum(MY);
  x01 = linspace(0,1-2^40*eps, 8*length(MX)).';
  
  %% extend periodically and method from CDT
  cdf0e = cdf0 + (-2:2); % extend periodically
  Xe = X + (-2:2);
  cdf1e = cdf1 + (-2:2);
  x01e = x01  + (-2:2);
  
  cdf0_inverse = interp1(cdf0e(:) + tc, Xe(:), x01e(:), 'linear', 'extrap');
  cdf1_inverse = interp1(cdf1e(:),      Xe(:), x01e(:), 'linear', 'extrap');
  
  cdt1 = interp1(cdf0_inverse, cdf0_inverse-cdf1_inverse, X, 'linear', 'extrap');
  cost = sum(cdt1(:).^2 .* MX(:));
  
  %% Inverse CDT for barycenter
  fe = Xe(:) - repmat(options.BaryWeight * cdt1, 5,1);
  fprime = gradient(fe) ./ gradient(Xe(:));
  bary = interp1(fe, repmat(MX,5,1)./fprime, X, 'linear', 'extrap');
  bary = bary/sum(bary );
end
