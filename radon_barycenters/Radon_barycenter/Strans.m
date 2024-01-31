classdef Strans < handle
  properties (SetAccess = protected)
    N = []  % degree of NFSFT, NFSOFT
    M_s2 = []  % number of points on the sphere
    M_s2_2 = []  % number of points on sphere in the range
    M_t = [] % number fo points on interval
    size_range = []  % vector of size of range
    y % Points on sphere (range)
    w2 = []  % weights on S2 (range)
    t
    w_t = []  % weights on t
    NFSFT = []  % Fourier transform on S2
    NFSFT2 = [] % Fourier transform on SO3 (for range)
    s2_gridtype = 'gl'
    PP = [] % quasi-eigenvalues lambda
    vol_range = 8 * pi
    ind_n % vector gives the degree n at position of spherical Fourier coefficients
  end
  properties (Dependent=true)
    w_range % same as w_so3
  end
  properties (SetAccess = public)
    x_s2 = []
    w_s2 = []
  end %properties

  methods
    function h = Strans(n, m, s2_grid) % Constructor
      h.N = n;
      h.M_s2 = m;
      if(nargin>2)
        h.s2_gridtype = s2_grid;
      end
      h.compute_range_grid;
      h.compute_sv; % depends on grid

      addpath('../nfft-3.5.3-matlab-openmp/nfsft/')
      h.NFSFT = nfsft(h.N, h.M_s2, NFSFT_NORMALIZED);
      h.NFSFT2 = nfsft(h.N, length(h.y), NFSFT_NORMALIZED);
      h.NFSFT2.x = h.y;
    end %function Arctrans

    function g = forward(h, fx)
      % compute spherical Fourier coefficients of f
      h.NFSFT.x = h.x_s2;
      h.NFSFT.f = fx(:) .* h.w_s2(:);
      nfsft_adjoint(h.NFSFT);
      fh = h.NFSFT.fhat.f_hat;

      g = zeros(h.size_range);

      for tj = 1:h.M_t
        h.NFSFT2.fhat = fh .* h.PP(:,tj);
        h.NFSFT2.nfsft_trafo;
        g(:,tj) = h.NFSFT2.f * 2 * pi;
      end
    end %function forward

    function fx_rec = inverse(h, gx)
      gh = zeros((h.N + 1)^2, h.M_t);
      for tj = 1:h.M_t
        h.NFSFT2.f = gx(:,tj) .* h.w2(:);
        h.NFSFT2.nfsft_adjoint;
        gh(:,tj) = h.NFSFT2.fhat.f_hat;
      end

      fh = sum(gh .* h.w_t(:).' .* h.PP .* (h.ind_n(:) + 1/2), 2);

      h.NFSFT.fhat = fh;
      h.NFSFT.nfsft_trafo;
      fx_rec = h.NFSFT.f * 2;
    end %function inverse

    function adj = adjoint(h, gx) % TODO: Normalization
      gh = zeros((h.N + 1)^2, h.M_t);
      for tj = 1:h.M_t
        h.NFSFT2.f = gx(:,tj) .* h.w2(:);
        h.NFSFT2.nfsft_adjoint;
        gh(:,tj) = h.NFSFT2.fhat.f_hat;
      end

      fh = sum(gh .* h.w_t(:).' .* h.PP, 2);

      h.NFSFT.fhat = fh;
      h.NFSFT.nfsft_trafo;
      adj = h.NFSFT.f;
    end %function inverse

    function set.x_s2(h,x)
      if (size(x,1)~=2) || (size(x,2) ~= h.M_s2)
        error('S2 nodes must be a 2xM matrix')
      end
      h.x_s2 = x;
    end

    function w = get.w_range(h)
      w = h.w2(:) .* h.w_t(:).';
    end
  end

  methods (Access=protected)
    function compute_sv(h)
      %% Use direct formula PP(i,j) = P_{n_i}(t_j)
      P = lepolym_new(h.N, h.t); % make t the rows
      h.ind_n = ceil(sqrt(1:(h.N+1)^2)).' - 1;
      h.PP = P(h.ind_n + 1, :);
    end %function compute_sv

    function compute_range_grid(h)
      switch h.s2_gridtype
        case 'gl' % Gauss-Legendre sampling
          [h.y, h.w2] = gl(h.N);
        case 'design'
          N_gl = 1 * h.N; % Exactness degree, usually 2 *N
          % Manuel
          if(N_gl<=4)
            qs2 = importdata('../quadrature/S2quads/N004_M10_C4.dat',' ',2);
          elseif(N_gl<=8)
            qs2 = importdata('../quadrature/S2quads/N008_M28_Tetra.dat',' ',2);
          elseif(N_gl<=12)
            qs2 = importdata('../quadrature/S2quads/N012_M58_C4.dat',' ',2);
          elseif(N_gl<=16)
            qs2 = importdata('../quadrature/S2quads/N016_M98_C4.dat',' ',2);
          elseif(N_gl<=20)
            qs2 = importdata('../quadrature/S2quads/N020_M148_Tetra.dat',' ',2);
          elseif(N_gl<=24)
            qs2 = importdata('../quadrature/S2quads/N024_M210_C4.dat',' ',2);
          elseif(N_gl<=28)
            qs2 = importdata('../quadrature/S2quads/N028_M282_C4.dat',' ',2);
          elseif(N_gl<=32)
            qs2 = importdata('../quadrature/S2quads/N032_M364_Tetra.dat',' ',2);
          elseif(N_gl<=36)
            qs2 = importdata('../quadrature/S2quads/N036_M458_C4.dat',' ',2);
          elseif(N_gl<=44)
            qs2 = importdata('../quadrature/S2quads/N044_M672_Ico.dat',' ',2);
          elseif(N_gl<=48)
            qs2 = importdata('../quadrature/S2designs/N048_M1200_Octa.dat',' ',2);
          elseif(N_gl<=52)
            qs2 = importdata('../quadrature/S2designs/N052_M1404_Tetra.dat',' ',2);
          elseif(N_gl<=56)
            qs2 = importdata('../quadrature/S2designs/N056_M1620_Tetra.dat',' ',2);
          elseif(N_gl<=60)
            qs2 = importdata('../quadrature/S2designs/N060_M1860_Tetra.dat',' ',2);
          elseif(N_gl<=64)
            qs2 = importdata('../quadrature/S2designs/N064_M2112_Ico.dat',' ',2);
          elseif(N_gl<=68)
            qs2 = importdata('../quadrature/S2designs/N068_M2376_Octa.dat',' ',2);
          elseif(N_gl<=72)
            qs2 = importdata('../quadrature/S2designs/N072_M2664_Octa.dat',' ',2);
          elseif(N_gl<=76)
            qs2 = importdata('../quadrature/S2designs/N076_M2964_Tetra.dat',' ',2);
          elseif(N_gl<=80)
            qs2 = importdata('../quadrature/S2designs/N080_M3276_Tetra.dat',' ',2);
          elseif(N_gl<=84)
            qs2 = importdata('../quadrature/S2designs/N084_M3612_Ico.dat',' ',2);
          elseif(N_gl<=88)
            qs2 = importdata('../quadrature/S2designs/N088_M3960_Octa.dat',' ',2);
          elseif(N_gl<=90)
            qs2 = importdata('../quadrature/S2designs/N090_M4140_Tetra.dat',' ',2);
          elseif(N_gl<=94)
            qs2 = importdata('../quadrature/S2designs/N094_M4512_Ico.dat',' ',2);
          elseif(N_gl<=98)
            qs2 = importdata('../quadrature/S2designs/N098_M4896_Octa.dat',' ',2);
          elseif(N_gl<=100)
            qs2 = importdata('../quadrature/S2designs/N100_M5100_Tetra.dat',' ',2);
          elseif(N_gl<=114)
            qs2 = importdata('../quadrature/S2designs/N114_M6612_Ico.dat',' ',2);
          elseif(N_gl<=124)
            qs2 = importdata('../quadrature/S2designs/N124_M7812_Ico.dat',' ',2);
          elseif(N_gl<=200)
            qs2 = importdata('../quadrature/S2designs_high/Design_21000_200_random.dat',' ',2);
          elseif(N_gl<=500)
            qs2 = importdata('../quadrature/S2designs_high/Design_130000_500_random.dat',' ',2);
          elseif(N_gl<=1000)
            qs2 = importdata('../quadrature/S2designs_high/Design_520000_1000_random.dat',' ',2);
          else
            error('Wrong degree for spherical design')
          end
          if(size(qs2.data,2)>3)
            h.w2 = qs2.data(:,4).';
          else
            h.w2 = ones(1,size(qs2.data,1))/size(qs2.data,1);
          end
          if (size(qs2.data,2) > 2) % Euclidean coordinates
            h.y=[atan2(qs2.data(:,1),qs2.data(:,2))';acos(qs2.data(:,3))'];
          else
            h.y = [qs2.data(:,1)'; (qs2.data(:,2))'];
          end
        otherwise
          error('Unknown quadrature rule')
      end

      [h.t, h.w_t] = lgwt(h.N + 1, -1,1);
      h.t = flip(h.t); % order ascending
      h.M_t = length(h.w_t);
      h.size_range = [length(h.w2), h.M_t];
    end %function compute_range_grid
  end %methods
end