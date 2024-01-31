% This Matlab file compares the Radon Wasserstein barycenters (or CDT interpolation)
% for different slicing approaches on the sphere.
% Requirements: 
%   NFFT matlab interface https://www-user.tu-chemnitz.de/~potts/nfft/download.php
%   Python installation with PythonOT https://pythonot.github.io
%   Quadrature formulas from https://www-user.tu-chemnitz.de/~potts/workgroup/graef/quadrature/

addpath ../nfft-3.5.3-matlab-openmp/nfsft/

N = 120;

% Nodes on sphere
M0 = N + 1;
ph0 = (0:2*N+1) / (2*N+2) * 2*pi;
[t_leg,w_t_leg] = lgwt(M0,-1,1);
[ph, th] = meshgrid(ph0,acos(t_leg));
w = w_t_leg(:) .* ones(size(ph0)) / numel(ph0) * 2*pi;
x = [ph(:).'; th(:).'];
N_phi = M0; N_the = 2 * M0 + 1;

% Add boundary nodes for plotting
ph1 = [ph0, ph0(1)];
th1 = [0; acos(t_leg(:)); pi];
X = cos(ph1).*sin(th1);
Y = sin(ph1).*sin(th1);
Z = cos(th1).*ones(size(ph1));
extend_nodes = @(f) [repmat(mean(f(1,:)), 1,size(f,2)+1); [f, f(:,1)]; repmat(mean(f(end,:)), 1,size(f,2)+1)];

% [X,w] = gl(N);
% ph = X(1,:);
% th = X(2,:);

% Radial Test Functions (symmetric in third component)
% fr = @(p,t, a,b) (sin(t) .* sin(b) .* cos(p-a) + abs(cos(t)).*cos(b));
fr = @(p,t, a,b) (sin(t) .* sin(b) .* cos(p-a) + (cos(t)).*cos(b));
% Von Mises-Fisher distributions
fx = exp(10 * fr(ph, th, 3.2, pi/2));
gx = exp(10 * fr(ph, th, 2.8, pi/2));

% Radial Test Functions
fr = @(p,t, a,b) (sin(t) .* sin(b) .* cos(p-a) + (cos(t)).*cos(b));
% Von Mises-Fisher distributions at north and south pole
fx = exp(10 * fr(ph, th, 3.2, 0.0));
gx = exp(10 * fr(ph, th, 5, pi));

% % Croissant measures
fx = (abs(ph - pi) < .314);
gx = (abs(ph - 5*pi/3) < .314);

% Quadratic spline (Smiley) (comment out)
f_r0=@(z,h) (z>h).*((z-h).^2./(1-h).^2);
x_0 = [0,pi; 0.5,2.2; 1.3,2.2; 0.9,1.5; 0.6,1.6; 1.2,1.6; ];
h_0 = [0.6; 0.98; 0.98; 0.9; 0.9; 0.9; ];
c_0 = [0.6; -0.7; -0.7; 0.4; 0.4; 0.4; ];
c_0 = [0.0; 0.7; 0.7; 0.4; 0.4; 0.4; ]; % must be >=0
%f_r = @(z,h)f_r0(-z,h)+f_r0(z,h); % f has to be even
f_r = @(z,h)f_r0(z,h);
g=@(ph,t)sum(c_0.*f_r(-(cos(x_0(:,2)).*t ...
    +sin(x_0(:,2)).*sqrt(1-t.^2).*cos(ph-x_0(:,1))),h_0),1);
gx = reshape(g(ph(:)', cos(th(:)')), size(ph));
% Radial Test Function
fr = @(p,t, a,b) (sin(t) .* sin(b) .* cos(p-a) + (cos(t)).*cos(b));
% Von Mises-Fisher distribution
fx = exp(5 * fr(ph, th, 4.2, 1.0));

% Normalize
fx = fx / sum(fx(:) .* w(:));
gx = gx / sum(gx(:) .* w(:));

figure(1)
surf(X, Y, Z, extend_nodes(fx), 'EdgeColor', 'none');
colorbar; axis equal; axis tight; set(gcf, 'Color', 'w'); % Sets axes background
xticklabels(''); yticklabels(''); zticklabels(''); 
title('Function f')

figure(2)
surf(X, Y, Z, extend_nodes(gx), 'EdgeColor', 'none');
colorbar; axis equal; set(gcf, 'Color', 'w'); 
xticklabels(''); yticklabels(''); zticklabels(''); 
title('Function g')

%% Create Vtrans and Arctrans instances
pv = Vtrans(N, numel(ph));
pv.x_s2 = x;
pv.w_s2 = w;

ps = Strans(N, numel(ph), 'design');
ps.x_s2 = x;
ps.w_s2 = w;

pa = Arctrans(N, numel(w), 'weighted', 's1s2');
pa.x_s2 = x;
pa.w_s2 = w;

% Forward transform
Vf = real(pv.forward(fx)) * 2*pi;
Vg = real(pv.forward(gx)) * 2*pi;

Vf(Vf<0) = 0;
Vg(Vg<0) = 0;

Sf = real(ps.forward(fx));
Sg = real(ps.forward(gx));

Sf(Sf<0) = 0;
Sg(Sg<0) = 0;

Af = pa.forward(fx);
Ag = pa.forward(gx);

%% Sliced barycenter
delta = 0.5;[.2 .4 .5 .6 .8]; % weight of barycenters

% CDT for Vertical Slice
Cf = cdt(real(Vf.'), pv.t, pv.w_t, pv.t, [-1,1]); % make t the first direction
Cg = cdt(real(Vg.'), pv.t, pv.w_t, pv.t, [-1,1]);

CSf = cdt(real(Sf.'), ps.t, ps.w_t, pv.t, [-1,1]); % make t the first direction
CSg = cdt(real(Sg.'), ps.t, ps.w_t, pv.t, [-1,1]);

S_dist = sqrt(sum((CSf - CSg).^2 .* w_t_leg .* ps.w2, "all"));

% Computation for A
Af = real(reshape(Af, pa.size_so3));
Ag = real(reshape(Ag, pa.size_so3));
Af(Af<0) = 0; % Af must be <=0 if f is
Ag(Ag<0) = 0;
Ah = zeros(size(Af));
tc = zeros(prod(pa.size_so3(2:end)), 1);
alpha = pa.alpha(:,1) / (2*pi); % Vector for compatibility with OT

for k = 1:length(delta)
  %% Vertical slice
  tic;
  Ch = delta(k) * Cf + (1 - delta(k)) * Cg;
  Vh = icdt(Ch, pv.t, pv.w_t);
  
  Vh = Vh.'; % reverse direction
  bary_V = reshape(real(pv.inverse(Vh)), size(fx));
  bary_V = bary_V / sum(bary_V(:).*w(:));
  time.v = toc;
  
  figure(4)
  surf(X, Y, Z, extend_nodes(bary_V), 'EdgeColor', 'none');
  colorbar; axis equal; set(gcf, 'Color', 'w'); 
  xticklabels(''); yticklabels(''); zticklabels(''); 
  title(sprintf('Barycenter Vertical %g', delta(k)))

%   figure(11)
%   imagesc(real(Vg)); colorbar
%   title('Vg')
%   figure(12)
%   imagesc(real(Vh)); colorbar
%   title('Vh')
%   figure(13)
%   imagesc(abs(Vg-Vh)); colorbar

  %% Parallel slice
  tic;
  CSh = delta(k) * CSf + (1 - delta(k)) * CSg;
  Sh = icdt(CSh, ps.t, ps.w_t);
  
  Sh = Sh.'; % reverse direction
  bary_S = reshape(real(ps.inverse(Sh)), size(fx));
  bary_S = bary_S / sum(bary_S(:).*w(:));
  time.s = toc;
  
  figure(5)
  surf(X, Y, Z, extend_nodes(bary_S), 'EdgeColor', 'none');
  colorbar; axis equal; set(gcf, 'Color', 'w'); 
  xticklabels(''); yticklabels(''); zticklabels(''); 
  title(sprintf('Barycenter Parallel %g', delta(k)))


  %% Semicircle
  tic
  for i = 1:prod(pa.size_so3(2:end))
    if k == 1
      [~,Ah(:,i),tc(i)] = ...
        OTcircle(alpha, real(Af(:,i)), real(Ag(:,i)), BaryWeight=delta(1));
    else
      [~,Ah(:,i),~] = ...
        OTcircle(alpha, real(Af(:,i)), real(Ag(:,i)), BaryWeight=delta(k), TC=tc(i));
    end % if
  end
  Ah = Ah * numel(alpha) / (2*pi); % circleOT normalizes to sum = 1
  
  % Barycenter on the sphere
  bary_A = reshape(real(pa.inverse(Ah)), size(fx));
  time.w = toc;
  
  figure(6)
  surf(X, Y, Z, extend_nodes(bary_A),'LineStyle','none')
  colorbar; axis equal; set(gcf, 'Color', 'w'); 
  xticklabels(''); yticklabels(''); zticklabels(''); 
  title(sprintf('Barycenter Semicircle %g', delta(k)))


  %% Comparison with Python-OT
  writematrix([fx(:), gx(:), w(:)],'fgw.txt');
  writematrix(x, 'x.txt')
  clear Af Ag Ah Sf Sg Sh CSf CSg CSh pa ps pv Cf Cg Ch PDparam Vf Vg Vh bary_pot bary_V
  
  % reg=0 means lp algorithm
  reg_pot = 0.01; % 0.1
  numItermax = 100;  % higher value did not change much
  tic
  system(sprintf('python pot_wrapper.py %g %g %d', delta(k), reg_pot, numItermax))
  time.pot = toc;
  
  bary_pot = readmatrix('bary_wass.txt');
  bary_pot = real(reshape(bary_pot,size(fx)));
  
  figure(10)
  surf(X, Y, Z, extend_nodes(bary_pot),'LineStyle','none')
  colorbar; axis equal; set(gcf, 'Color', 'w'); 
  xticklabels(''); yticklabels(''); zticklabels(''); 
  title(sprintf('Barycenter POT %g', delta(k)))
  
end
