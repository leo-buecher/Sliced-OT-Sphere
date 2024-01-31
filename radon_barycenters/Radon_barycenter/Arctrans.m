classdef Arctrans < handle
  properties (SetAccess = protected)
    N = []  % degree of NFSFT, NFSOFT
    M_s2 = []  % number of points on the sphere
    M_so3 = []  % number of points on SO3
    size_so3 = []  % vector of length of Euler angles alpha,beta,gamma
    alpha % Euler angles of SO3 grid
    beta
    gamma
    w_so3 = []  % weights on SO3
    NFSFT = []  % Fourier transform on S2
    NFSOFT = [] % Fourier transform on SO3
    transform = 'weighted' % 'unweighted' is currently not compatible with OT
    so3_gridtype = 'gl'
    PP = [] % quasi-eigenvalues lambda
    mu = [] % computed singular values of arc transform
  end
  properties (Dependent=true)
    size_range % same as size_so3
    w_range % same as w_so3
    vol_range % 8pi^2
  end
  properties (SetAccess = public)
    x_s2 = []
    w_s2 = []
  end %properties

  methods
    function h = Arctrans(n, m, transform, so3_grid) % Constructor
      h.N = n;
      h.M_s2 = m;
      if(nargin>2)
        h.transform = transform;
      end
      if(nargin>3)
        h.so3_gridtype = so3_grid;
      end
      h.PP = compute_sv(h);
      h.compute_so3_grid;

      addpath('../nfft-3.5.3-matlab-openmp/nfsft/')
      addpath('../nfft-3.5.3-matlab-openmp/nfsoft/')
      h.NFSFT = nfsft(h.N, h.M_s2, NFSFT_NORMALIZED);
      flags = NFSOFT_REPRESENT + NFSFT_USE_DPT;
      n_nfft = ceil(3 * h.N / 2) * 2; % 4*h.N+8;
      h.NFSOFT = nfsoft(h.N, prod(h.size_so3), flags, 0, 6, 1000, n_nfft);
      h.NFSOFT.x = [h.alpha(:)'; h.beta(:)'; h.gamma(:)'];
    end %function Arctrans

    function g = forward(h, fx)
      % compute spherical Fourier coefficients of f
      h.NFSFT.x = h.x_s2;
      h.NFSFT.f = fx(:) .* h.w_s2(:);
      nfsft_adjoint(h.NFSFT);
      fh = double(h.NFSFT.fhat);

      % multiply Fourier coefficients with singular values
      gh = zeros(nfsoft_f_hat_size(h.N),1);
      for n=0:h.N
        P = h.PP(h.N+1+(-n:n),n+1);     % (2n+1)*1
        g = fh(h.N+1+(-n:n),n+1) .* P.';
        gh(nfsoft_f_hat_size(n-1)+1:nfsoft_f_hat_size(n-1)+(2*n+1)^2) = g(:);
      end
      
      h.NFSOFT.fhat = gh(:);
      nfsoft_trafo(h.NFSOFT);
      g = h.NFSOFT.f;
    end %function forward

    function fx_rec = inverse(h, gx)
      fh_inverted = zeros(size(h.NFSFT.fhat));

      h.NFSOFT.f = gx(:) .* h.w_so3(:);
      nfsoft_adjoint(h.NFSOFT);
      gh_inverted = h.NFSOFT.fhat;
      h.mu = zeros(h.N+1,1);  % singular values
      for n=0:h.N
        gh_inverted(nfsoft_f_hat_size(n-1)+1:nfsoft_f_hat_size(n-1)+(2*n+1)^2,1) ...
          =(2*n+1)*gh_inverted(nfsoft_f_hat_size(n-1)+1:nfsoft_f_hat_size(n-1)+(2*n+1)^2,1);
        P = h.PP(h.N+1+(-n:n),n+1);
        h.mu(n+1) = sqrt(sum(P.^2)); % * 4*pi / sqrt(2*n+1);
        Ps = 1./P;
        Ps(2:2:end) = 0;
        if n>1; Ps(n+1) = 0; end % central entry also vanishes
        g = reshape(gh_inverted(nfsoft_f_hat_size(n-1)+1:nfsoft_f_hat_size(n-1)+(2*n+1)^2,1)...
            , 2*n+1,2*n+1);
%         fh_inverted(h.N+1+(-n:n),n+1) = g * Ps /sum(abs(Ps)>0);
        fh_inverted(h.N+1+(-n:n),n+1) = g * P /h.mu(n+1)^2; %*(1-(n/(h.N+1))^2);
      end
      
      h.NFSFT.fhat = fh_inverted;
      nfsft_trafo(h.NFSFT);
      fx_rec = h.NFSFT.f;
    end %function inverse

    function fx_rec = adjoint(h, gx)
      fh_inverted = zeros(size(h.NFSFT.fhat));

      h.NFSOFT.f = gx(:) .* h.w_so3(:);
      nfsoft_adjoint(h.NFSOFT);
      gh_inverted = h.NFSOFT.fhat;
      h.mu = zeros(h.N+1,1);  % singular values
      for n=0:h.N
        gh_inverted(nfsoft_f_hat_size(n-1)+1:nfsoft_f_hat_size(n-1)+(2*n+1)^2,1) ...
          =(2*n+1)*gh_inverted(nfsoft_f_hat_size(n-1)+1:nfsoft_f_hat_size(n-1)+(2*n+1)^2,1);
        P = h.PP(h.N+1+(-n:n),n+1);
        h.mu(n+1) = sqrt(sum(P.^2)); % * 4*pi / sqrt(2*n+1);
        Ps = 1./P;
        Ps(2:2:end) = 0;
        if n>1; Ps(n+1) = 0; end % central entry also vanishes
        g = reshape(gh_inverted(nfsoft_f_hat_size(n-1)+1:nfsoft_f_hat_size(n-1)+(2*n+1)^2,1)...
            , 2*n+1,2*n+1);
        fh_inverted(h.N+1+(-n:n),n+1) = g * P; % /h.mu(n+1)^2; %*(1-(n/(h.N+1))^2);
      end
      
      h.NFSFT.fhat = fh_inverted;
      nfsft_trafo(h.NFSFT);
      fx_rec = h.NFSFT.f;
    end %function adjoint

    function Af = forward_direct(h, f, quad_pts)
      if(nargin<3)
        quad_pts = 2*h.N;
      end
      Af = zeros(numel(h.alpha),1);%zeros(eval_pts,eval_pts,eval_pts,eval_pts);
      tq = linspace(-1,1,quad_pts);
      switch h.transform
        case 'weighted'
          for l = 1:length(Af)
            ph = mod(atan2(...
              sqrt(1-tq.^2)*(-cos(h.alpha(l))*cos(h.beta(l))*sin(h.gamma(l))-sin(h.alpha(l))*cos(h.gamma(l)))+sin(h.beta(l))*tq*sin(h.gamma(l)),...
              sqrt(1-tq.^2)*(cos(h.alpha(l))*cos(h.beta(l))*cos(h.gamma(l))-sin(h.alpha(l))*sin(h.gamma(l)))-sin(h.beta(l))*tq*cos(h.gamma(l))...
              ),2*pi);
            tt = cos(h.alpha(l))*sin(h.beta(l))*sqrt(1-tq.^2)+cos(h.beta(l))*tq;
            Af(l) = sum(f(ph,tt),2)/quad_pts.*2;
          end
        case 'unweighted'
          for l=1:length(Af)
            quad_nodes = (((1:quad_pts)-1/2)/quad_pts*2-1).*t(l_t);%linspace(-t,t,quad_pts);
            ph=mod(atan2(...
              -cos(h.beta(l)).*sin(h.gamma(l)).*cos(h.alpha(l)-quad_nodes)-cos(h.gamma(l)).*sin(h.alpha(l)-quad_nodes),...
              sin(h.alpha(l)).*(cos(h.beta(l)).*cos(h.gamma(l)).*sin(quad_nodes)-sin(h.gamma(l)).*cos(quad_nodes))+...
              cos(h.alpha(l)).*(cos(h.beta(l)).*cos(h.gamma(l)).*cos(quad_nodes)+sin(h.gamma(l)).*sin(quad_nodes))...
              ),2*pi);
            tt=sin(h.beta(l)).*cos(h.alpha(l)-quad_nodes);
            % xi(ph,tt) = Q(-c,-b,-a) * e_t
            Af(l) = sum(f(ph,tt),2)/quad_pts.*t(l_t)*2;
          end
      end
    end %function forward_direct

    function set.x_s2(h,x)
      if (size(x,1)~=2) || (size(x,2) ~= h.M_s2)
        error('S2 nodes must be a 2xM matrix')
      end
      h.x_s2 = x;
    end

    function sz = get.size_range(h)
      sz = h.size_so3;
    end

    function w = get.w_range(h)
      w = h.w_so3;
    end

    function w = get.vol_range(h)
      w = 8*pi^2;
    end
  end

  methods (Access=protected)
    function PP = compute_sv(h)
      switch(h.transform)
        case 'weighted'
          %% Use direct formula with double factorials (iterative)
          PP = zeros(h.N+1,h.N+1);
          PP(1,1) = 2;
          PP(2,2) = +pi/2 / sqrt(2); % Theory says minus, maybe normalization in nfft
          for j = 1:h.N
            if (j>1)
              PP(j+1,j+1) = j/(j+1) * (2*j-1) * (2*j-3) * PP(j-1,j-1) ...
                / sqrt(2*j*(2*j-1)*(2*j-2)*(2*j-3));
            end
            for n = j+2:2:h.N
              PP(j+1,n+1) = (n-2) * (n+j-1) / (n-j) / (n+1) * PP(j+1,n-1) ...
                * sqrt((n-j) * (n-1-j) / (n+j) / (n-1+j)); 
            end
          end
          PP = PP .* sqrt((2*(0:h.N)+1) / (4*pi));
          PP = [flipud(PP(2:end,:)); PP];
        case 'unweighted'
          % calculate associated Legendre at 0
          plan_nfsft1 = nfsft_init_advanced(h.N,1,NFSFT_NORMALIZED);
          nfsft_set_x(plan_nfsft1,[0; pi/2]);
          nfsft_precompute_x(plan_nfsft1);
          nfsft_set_f(plan_nfsft1,1);
          nfsft_adjoint(plan_nfsft1);
          PP = nfsft_get_f_hat(plan_nfsft1);
          nfsft_finalize(plan_nfsft1)
          PP = real(PP);
        otherwise
          error('Unknown transform type')
      end
    end %function compute_sv

    function compute_so3_grid(h)
      switch h.so3_gridtype
        case 'gl' % Gauss-Legendre sampling
          N_gl = h.N; % degree of exactness (was 2*h.N in first paper)
          h.size_so3 = [2*N_gl+1, N_gl+1, 2*N_gl+1];
          [betac, wb] = lgwt(N_gl+1,-1,1);
          [alpha0,betac,gamma0] = ndgrid(0:2*N_gl,betac,0:2*N_gl);
          h.alpha = alpha0*2*pi/(2*N_gl+1);
          h.beta  = acos(betac);
          h.gamma = gamma0*2*pi/(2*N_gl+1);
          h.w_so3 = repmat(reshape(wb,1,[]),...
            2*N_gl+1, 1, 2*N_gl+1)/2/(2*N_gl+1)^2;    % sum=1
        case 's1s2'
          N_gl = 1 * h.N; % Exactness degree, usually 2 *N
          N_alpha = 2 * h.N; % exactness degree of alpha quadrature (2N_s+1 points)
          % Manuel_S1 x S2
          if(N_gl<=16)
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
              qs2 = importdata('../quadrature/S2designs_high/Design_5200_100_random.dat',' ',2);
          else
              error('Wrong degree')
          end
          if(size(qs2.data,2)>3)
              wb = qs2.data(:,4).';
          else
              wb = ones(1,size(qs2.data,1))/size(qs2.data,1);
          end
          if (size(qs2.data,2) > 2) % Euclidean coordinates
            qs2=[atan2(qs2.data(:,1),qs2.data(:,2))';acos(qs2.data(:,3))'];
          else
            qs2 = [qs2.data(:,1)'; (qs2.data(:,2))'];
          end
          
          h.gamma = mod(repmat(qs2(1,:), 2*N_alpha+1, 1), 2*pi);
          h.beta  = repmat(qs2(2,:), 2*N_alpha+1, 1);
          h.alpha = repmat((0:2*N_alpha).' / (2*N_alpha+1) * 2*pi, 1, length(wb));
          h.w_so3 = repmat(wb, 2*N_alpha+1, 1) * 2*pi/(2*N_alpha+1)  /2/pi;   % sum=1
          h.size_so3 = [2*N_alpha+1, numel(wb)];
        otherwise
          error('Unknown quadrature rule')
      end
    end %function compute_so3_grid
  end %methods
end