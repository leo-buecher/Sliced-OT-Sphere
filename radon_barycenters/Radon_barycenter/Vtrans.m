classdef Vtrans < handle
  properties (SetAccess = protected)
    N = []  % degree of NFSFT, NFSOFT
    M_s2 = []  % number of points on the sphere
    s % angles on Möbius strip
    t % length variable on Möbius strip
    w_t = []  % weights on t
    NFSFT = []  % Fourier transform on S2
    st_gridtype = 'gl'
    sv = [] % eigenvalues
    svi = [] % inverse eigenvalues
    P = [] % matrix of Legendre polynomials evaluated at t
  end
  properties (Dependent=true)
    size_range
    w_range % weights on the range
    vol_range % 2 * 2pi
  end
  properties (SetAccess = public)
    x_s2 = []
    w_s2 = []
  end %properties

  methods
    function h = Vtrans(n, m, ~, so3_grid) % Constructor
      h.N = n;
      h.M_s2 = m;
      if(nargin>3)
        h.so3_gridtype = so3_grid;
      end
      h.compute_st_grid;
      h.compute_sv;

      addpath('../nfft-3.5.3-matlab-openmp/nfsft/')
      h.NFSFT = nfsft(h.N, h.M_s2, NFSFT_NORMALIZED);
    end %function Arctrans

    function g = forward(h, fx)
      % compute spherical Fourier coefficients of f
      h.NFSFT.x = h.x_s2;
      h.NFSFT.f = fx(:) .* h.w_s2(:);
      nfsft_adjoint(h.NFSFT);
      fh = double(h.NFSFT.fhat);
      
      % multiply Fourier coefficients with singular values
      Vfh = double(fh) .* h.sv;
      g = (length(h.s) * ifft(Vfh) .* exp(-1i*h.N*h.s')) * h.P';
    end %function forward

    function fx_rec = inverse(h, gx)
      gh = (gx .* h.w_t.') * (h.P .* ((0:h.N)+1/2));
      gh = fft(gh .* exp(1i*h.N*h.s')) / length(h.s);
      
      h.NFSFT.fhat = h.svi.*gh;
      nfsft_trafo(h.NFSFT);
      fx_rec = h.NFSFT.f;
    end %function inverse

    function fx_rec = adjoint(h, gx)
      gh = (gx .* h.w_t.') * (h.P .* ((0:h.N)+1/2));
      gh = fft(gh .* exp(1i*h.N*h.s')) / length(h.s);
      
      h.NFSFT.fhat = gh .* h.sv; % only difference to inverse
      nfsft_trafo(h.NFSFT);
      fx_rec = h.NFSFT.f;
    end %function adjoint

    function g = forward_direct(h, fx)
      f_interp = griddedInterpolant(h.x_s2(1,:), h.x_s2(2,:), fx);
      [ss,tt] = meshgrid(h.s,h.t);
      
      phi = (1:h.N) * 2*pi / h.N;
      g = zeros(size(ss));

      for k = 1:length(phi)
        X = tt.*cos(ss)-sqrt(1-tt.^2).*cos(phi(k)).*sin(ss);
        Y = tt.*sin(ss)+sqrt(1-tt.^2).*cos(phi(k)).*cos(ss);
        Z = sqrt(1-tt.^2).*sin(phi(k));
        g = g + f_interp(atan2(Y,X), acos(Z) );
      end
      g = g / length(phi);
    end %function forward_direct

    function set.x_s2(h,x)
      if (size(x,1)~=2) || (size(x,2) ~= h.M_s2)
        error('S2 nodes must be a 2xM matrix')
      end
      h.x_s2 = x;
    end

    function sz = get.size_range(h)
      sz = [length(h.s), length(h.t)];
    end

    function w = get.w_range(h)
      w = ones(size(h.s(:))) .* h.w_t.' / numel(h.s) * 2*pi;
    end

    function w = get.vol_range(h)
      w = 4*pi;
    end
  end

  methods (Access=protected)
    function compute_sv(h)
      % Compute singular values of vertical slice transform
      h.sv = zeros(h.N+1);
      h.sv(1,1) = legendre(0,0,'norm')/sqrt(2*pi);
      for n=1:h.N
          h.sv(1:n+1,n+1) = legendre(n,0,'norm')/sqrt(2*pi);
      end
      h.sv = [flipud(h.sv); h.sv(2:h.N+1,:)];
      
      h.svi = zeros(size(h.sv));   
      h.svi(h.sv~=0) = 1./h.sv(h.sv~=0); % inverse singular values
      
      h.P = lepolym_new(h.N,h.t).'; % Legendre polynomial
    end %function compute_sv

    function compute_st_grid(h)
      switch h.st_gridtype
        case 'gl' % Gauss-Legendre sampling

          % Nodes on Möbius strip
          L0 = 2*h.N + 1;
          M0 = h.N + 1;
          h.s = 2 * pi / L0 * (0:L0-1);
          [h.t,h.w_t] = lgwt(M0,-1,1);
          h.t = flip(h.t); % order ascending
        otherwise
          error('Unknown quadrature rule')
      end
    end %function compute_st_grid
  end %methods
end