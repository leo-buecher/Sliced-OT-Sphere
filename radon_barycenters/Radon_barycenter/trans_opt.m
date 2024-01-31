% Author: J Salomon.
% A Matlab implementation of the algorithm proposed in the preprint of J. Delon, J. Salomon and A. Sobolevski (2009), see the publications page.
% [1] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
% https://users.mccme.ru/ansobol/otarie/software.html
% Last edit: M. Quellmalz (2023)

function [cout, tc] = trans_opt(lambda, X, Y, MX, MY)
  n0 = length(X);
  n1 = length(Y);
  
  CX				= cumsum(MX) ;
  CY				= cumsum(MY) ;
  
  Lm	 			= 10 ;
  Lp				= 10 ;
  L				= max(Lp,Lm) ;
	  
  epsi				= 1e-15 ;
	  
  tm				= -1 ;
  tp				=  1 ;
  tc				= .5*(tm+tp) ;	
	  
  pasfini				= 1 ;
  iter				= 0 ;
  
  while pasfini && iter < 100
	  iter			= iter + 1 ;
	  [dCp,dCm]		= dC(tc,X,Y,CX,CY,n0,n1,lambda) ; 
	  pasfini			= (dCp*dCm)>0 ;
      
      
	  if ((tp-tm)<epsi/L)*pasfini
		  [dCptp,dCmtp]	= dC(tp,X,Y,CX,CY,n0,n1,lambda) ;
		  [dCptm,dCmtm]	= dC(tm,X,Y,CX,CY,n0,n1,lambda) ;
		  Ctp		=  C(tp,X,Y,CX,CY,n0,n1,lambda) ;
		  Ctm		=  C(tm,X,Y,CX,CY,n0,n1,lambda) ;
		  if abs(dCptm-dCmtp)>0.001
			  tc	= (Ctp-Ctm + dCptm*tm - dCmtp*tp)/(dCptm-dCmtp) ;
		  end
		  pasfini		= 0 ;
    
      elseif (pasfini>0)
		    if dCp<0	
			    tm	= tc ;
		    else
			    tp	= tc ;
		    end	
		    tc		= .5*(tm+tp) ;
	  end
  end % while
  
  cout=C(tc,X,Y,CX,CY,n0,n1,lambda);
end % function

function valC = C(t,X,Y,CX,CY,n0,n1,lambda)
  Ip				= CY-(t-floor(t))>=0 ;
  In				= CY-(t-floor(t))< 0 ;
  
  valF0				= CX ;
  valF1t				= ([CY(Ip)-(t-floor(t)),CY(In)-(t-floor(t))+1]) ;
%   valF0				= ones(n0+n1,n0)*diag(CX) ;
%   valF1t				= ones(n0+n1,n1)*diag([CY(Ip)-(t-floor(t)),CY(In)-(t-floor(t))+1]) ;
  vsort				= [0,sort([valF0(1,:),valF1t(1,:)])] ;
  
  Yt				= [ Y(Ip)+floor(t) ,  Y(In)+1+floor(t) ] ;
  Yt				= [ Yt, Yt(1)+1] ; 
  
  vk0 				= (.5*(vsort(2:n0+n1+1)+vsort(1:n0+n1))).' ;
  vk1 				= (.5*(vsort(2:n0+n1+1)+vsort(1:n0+n1))).' ;
%   vk0 				= diag(.5*(vsort(2:n0+n1+1)+vsort(1:n0+n1)))*ones(n0+n1,n0) ;
%   vk1 				= diag(.5*(vsort(2:n0+n1+1)+vsort(1:n0+n1)))*ones(n0+n1,n1) ;
  
  [~,nxk]			= max(vk0< valF0 ,[] , 2  ) ; 
  [~,nyk]			= max(vk1 < [valF1t,1] ,[] , 2) ;
%   [vxk,nxk]			= max(vk0< valF0 ,[] , 2  ) ; 
%   [vyk,nyk]			= max([vk1,vk1(:,1)]<[valF1t,ones(n0+n1,1)] ,[] , 2) ;
  xk				= reshape(X (nxk), n0+n1 , 1 ) ; 
  yk				= reshape(Yt(nyk), n0+n1 , 1 ) ;
  size(xk);
  size(yk);
  valC  				= [vsort(2:n0+n1+1)-vsort(1:n0+n1)]*cout(xk,yk,lambda) ;
end

function [valdcp,valdcm] = dC(t,X,Y,CX,CY,n0,n1,lambda)
  Ip				= CY-(t-floor(t))>=0 ;
  In				= CY-(t-floor(t))< 0 ;
  
  valF0				= CX; %MQ
  valF1t				= [CY(Ip)-(t-floor(t)),CY(In)-(t-floor(t))+1].' ; %MQ
%   valF0				= ones(n1,n0)*diag(CX) ;
%   valF1t				= diag([CY(Ip)-(t-floor(t)),CY(In)-(t-floor(t))+1])*ones(n1,n0) ;
  vsort				= [0,sort([valF0(1,:),[valF1t(:,1)]'])] ;
  diff				= .5*(abs(vsort(2:n0+n1+1) - vsort(1:n0+n1))) ;
  epsilon				= min(diff(diff>0)) ;
  
  X				= [ X , X(1)+1] ;
  Yt				= [ Y(Ip)+floor(t) , Y(In)+1+floor(t) ] ;
  Yt				= [ Yt, Yt(1)+1] ;
  
  [~,nxk]			= max( valF1t               <=   valF0                      , [] , 2) ;
  [~,nxkm]			= max(valF1t  <= ([valF0-epsilon,1]) , [] , 2) ;
%   [vxk,nxk]			= max( valF1t               <=   valF0                      , [] , 2) ;
%   [vxkm,nxkm]			= max([valF1t,valF1t(:,1)]  <= ([valF0-epsilon,ones(n1,1)]) , [] , 2) ;
  xk				= X (nxk' ) ;
  xkm				= X (nxkm') ;
  
  valdcp				= sum(cout(xk ,Yt(2:n1+1),lambda) - cout(xk ,Yt(1:n1),lambda)) ;
  valdcm				= sum(cout(xkm,Yt(2:n1+1),lambda) - cout(xkm,Yt(1:n1),lambda)) ;
end

function res=cout(x,y,lambda)
  res				= (abs(x-y)).^lambda ;
end
