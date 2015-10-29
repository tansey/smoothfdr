function initials=initialvalue(signalfile,zvaluefile,reglocafile,L)
%%L is number of the components in the normal mixture



signal=load(signalfile);%%%% signals, of the brain region of interest (ROI), gotten by BH procedure 
zvalue=load(zvaluefile);%%%% observed data on the whole image grid including the brain part and the non-brain part
regloca=load(reglocafile);%%%%% location of the ROI on the LX*LY*LZ image grid
 


n_reg=length(regloca);% the number of voxels in the ROI
zvalue_1=zeros(n_reg,1);% to store the z-value of the signals by BH procedure
num=0;
for i=1:n_reg
   pi=regloca(i);
   if signal(i)==1
     num=num+1;
     zvalue_1(num)=zvalue(pi);
     
   end
end
prop=1-num/n_reg;

%%%%%%%%%%%always use set the initial values for beta and h to be zeros
beta=0;
h=0;

initials=[beta,h];


temp=0;



try
  obj=gmdistribution.fit(zvalue_1(1:num,1),L);%% gmdistribution.fit is a Matlab built-in function, see http://www.mathworks.com/help/stats/gmdistribution.fit.html
  for j=1:L
      initials=[initials,obj.mu(j),obj.Sigma(j),obj.PComponents(j)];
  end
  
  
catch err %%%%%%% if gmdistribution.fit is divergent 
  mu=zeros(L,1);

  if L==1
     mu(1)=1;

  elseif mod(L,2)==0 %%%%%%if L is divisible by 2, then let mu_i be [-L,-L/2,-L/4,...,L/4,L/2,L] and thus symmetric around 0.
    temp=L/2;
    for mu_i=1:temp
      mu(mu_i)=mu_i*2;
      mu(L-mu_i+1)=-mu(mu_i);
    end
  else  %%%%%%%if L is not divisible by 2, then let mu_i be [-floor(L/2),-floor(L/2)+1,...,0,floor(L/2)-1,floor(L/2)] and thus symmetric around 0.
    temp=ceil(L/2);
      for mu_i=1:L
        mu(mu_i)=mu_i-temp;
      end

  end



  Sigma=ones(L,1);
  PComponents=(1/L)*ones(L,1);
  
  for j=1:L
      initials=[initials,mu(j),Sigma(j),PComponents(j)];
  end

end
