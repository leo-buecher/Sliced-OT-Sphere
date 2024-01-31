function [varargout]=lepolym_new(n,x)
  
% lepolym  Legendre polynomials of degree up to n, i.e., 0,1, ...,n
    % y=lepolym(n,x) returns the Legendre polynomials
    % The degree should be a nonnegative integer 
    % The argument x is a vector be on the closed interval [-1,1]; 
    % [dy,y]=lepolym(n,x) also returns the values of 1st-order 
    %  derivatives of the Legendre polynomials upto n stored in dy
    % Note: y (and likewise for dy) saves L_0(x), L_1(x), ...., L_n(x) by rows
    % i.e., L_k(x) is the (k+1)th row of the matrix y (or dy)
% Last modified on July 18, 2014    
      
dim=size(x); xx=x; if dim(1)>dim(2), xx=xx'; end; % xx is a row-vector
if nargout==1,
     if n==0, varargout{1}=ones(size(xx));  return; end;
     if n==1, varargout{1}=[ones(size(xx));xx]; return; end;
     y=zeros(n,size(xx,2));
     y(1,:)=ones(size(xx)); 
     y(2,:)=xx;   % L_0(x)=1, L_1(x)=x
     for  k=2:n,                      % Three-term recurrence relation:  
	   y(k+1,:)=((2*k-1)*xx.*y(k,:)-(k-1)*y(k-1,:))/k; % kL_k(x)=(2k-1)xL_{k-1}(x)-(k-1)L_{k-2}(x)
     end;
     varargout{1}=y;
end;

if nargout==2,
     if n==0, varargout{2}=ones(size(xx)); varargout{1}=zeros(size(xx)); return;end;
     if n==1, varargout{2}=[ones(size(xx));xx]; 
              varargout{1}=[zeros(size(xx));ones(size(xx))]; return; end;

     polylst=ones(size(xx)); pderlst=zeros(size(xx));poly=xx; pder=ones(size(xx));
     y=[polylst;poly]; dy=[pderlst;pder];
     % L_0=1, L_0'=0, L_1=x, L_1'=1
    for k=2:n,                          % Three-term recurrence relation:  
      polyn=((2*k-1)*xx.*poly-(k-1)*polylst)/k; % kL_k(x)=(2k-1)xL_{k-1}(x)-(k-1)L_{k-2}(x)
      pdern=pderlst+(2*k-1)*poly;  % L_k'(x)=L_{k-2}'(x)+(2k-1)L_{k-1}(x)
 	  polylst=poly; poly=polyn; y=[y;polyn];
	  pderlst=pder; pder=pdern; dy=[dy;pdern];
    end;
      varargout{2}=y; varargout{1}=dy;
end;

return
      

