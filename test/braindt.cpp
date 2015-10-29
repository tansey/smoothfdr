#include"braindt.h"
//#include <iostream>

//#include<vector>
#include <cfloat>//for IsFiniteNumber and IsNumber function
#include <algorithm>
#include<cstdio>
#include<cstdlib>
#include<string>
//#include<cstring>
#define _USE_MATH_DEFINES
#include<math.h>
using namespace std;
const double braindt::TINY=1e-20;
const double braindt::mu0=0;
const double braindt::sigma0_sq=1;
//constructor function of class braindt 
braindt::braindt(int &LX, int &LY, int &LZ,int &Num,char *argv ,vector<int> &Loca,int &PN,vector<double>& est,int &L_hat,int sweep_b, int sweep_r,int sweep_lis,int iter_max):\
	Lx(LX),Ly(LY),Lz(LZ),N(Num),pN(PN),y(PN),L(L_hat),U(2),invI(3),I(3),swp_b(sweep_b),swp_r(sweep_r),max_iter(iter_max),initial(PN ),delta(2),\
    H_mean0(2),H0(sweep_r,vector<double>(2)),H_cond_mean(2),mu1_hat_pre(L_hat ),sigma1_sq_hat_pre(L_hat ),pEll_hat_pre(L_hat ),LIS(PN),swp_lis(sweep_lis)
{
       
    
    
   
  

	vector<double> y_ori;//observed data on the whole image grid including the brain part and the non-brain part
	ifstream ifs(argv);//
	double temp_y;
	while(ifs>>temp_y)
		y_ori.push_back(temp_y);// read in y_ori
    
    
    
	
	int i;
        loca=Loca.begin();//the location of the brain region of interest (ROI) on the Lx*Ly*Lz image grid

	for(int pi=0;pi<pN;pi++)
	{
		i=loca[pi];
		y[pi]=y_ori[i];//the observed data of the ROI
		
	}
	

	YbyX=Ly*Lx;

	beta_hat=est[0];
	h_hat=est[1];

    int i3;
	for(int i=0;i<L;i++)
	{
        i3=i*3;
		mu1_hat.push_back(est[2+i3]);
		sigma1_sq_hat.push_back(est[3+i3]);
		pEll_hat.push_back(est[4+i3]);
		
	}
	
	
}

inline double braindt::normpdf(double &x, double &mu, double &sigma_sq)// The probability density function of the normal distribution with mean=mu, variance=sigma_sq

{
    return exp(-1*(x-mu)*(x-mu)/(2*sigma_sq))/(sqrt(2*M_PI*sigma_sq));
}


inline bool braindt::IsFiniteNumber(double &x)// Judge whether x is a finite numebr
{
	return (x<=DBL_MAX && x>=-DBL_MAX);
}        
inline bool braindt::IsNumber(double &x)//Judge whether x is a numebr
{
	return (x==x);
}

double braindt::sigmasum(vector<double> &p, int begin, int end)//sum of elements of vector p
{
	double sum=0;
	for(int i=begin; i<end; i++)
	{
		sum+=p[i];
	}
	return sum;
}
double braindt::sigmasum(vector<double> &p1, vector<double> &p2, int begin, int end)//sum of elements of the entrywise-product of vectors p1 and p2
{
	double sum=0;
	for(int i=begin; i<end; i++)
	{
		sum+=p1[i]*p2[i];
	}
	return sum;
}
double braindt::sigmasum(vector<double> &p1, vector<double> &p2, double p2minus, int begin, int end)//sum of elements of the entrywise-product of vectors p1 and p3 squared, where p3=p2-p2minus
{
        double sum=0;
        double minus;
        for(int i=begin; i<end; i++)
        {
                minus=p2[i]-p2minus;
                sum+=p1[i]*minus*minus;
        }
        return sum;
        
}


void braindt::two_d_var_matrix_inv(ofstream &fileout, int splnumber)// compute the inverse matrix of a 2 by 2 matrix
{
	
	double s1,s2;
	
	vector<double> var(3);
	
	for(int i=0;i<splnumber;i++)
	{
		s1=H[i][0]-H_mean[0];
		var[0]+=s1*s1;
		s2=H[i][1]-H_mean[1];
		var[2]+=s2*s2;
		var[1]+=s1*s2;//non-diagonal entry
		
	
		
	}
	
	
	
	splnumber-=1;
	var[0]/=splnumber;
	var[2]/=splnumber;
	var[1]/=splnumber;
        for(int t=0;t<3;t++)
        {fileout<<"I["<<t<<"]="<<var[t]<<endl;}
		
	double denom=var[0]*var[2]-var[1]*var[1];
	if(denom>0) 
	{
		invI[0]=var[2]/denom;
		invI[2]=var[0]/denom;
		invI[1]=-var[1]/denom;
		if(!(IsFiniteNumber(invI[0])&&IsFiniteNumber(invI[1])&&IsFiniteNumber(invI[2])&&IsNumber(invI[0])&&IsNumber(invI[1])&&IsNumber(invI[2])))
		{
			fileout<<"invI is almost non-invertible matrix"<<endl;
			var[0]+=TINY;
			var[2]+=TINY;
			denom=var[0]*var[2]-var[1]*var[1];
			invI[0]=var[2]/denom;
			invI[2]=var[0]/denom;
			invI[1]=-var[1]/denom;
			if(!(IsFiniteNumber(invI[0])&&IsFiniteNumber(invI[1])&&IsFiniteNumber(invI[2])&&IsNumber(invI[0])&&IsNumber(invI[1])&&IsNumber(invI[2])))
			{
				fileout<<"we choose invI=I"<<endl;
                invI[0]=1;
                invI[1]=0;
                invI[2]=1;
                
			}
			
			
		}
		
		
		
	}
	else {
		fileout<<"invI is non-invertible matrix"<<endl;
		var[0]+=TINY;
		var[2]+=TINY;
		denom=var[0]*var[2]-var[1]*var[1];
		invI[0]=var[2]/denom;
		invI[2]=var[0]/denom;
		invI[1]=-var[1]/denom;
		if(!(IsFiniteNumber(invI[0])&&IsFiniteNumber(invI[1])&&IsFiniteNumber(invI[2])&&IsNumber(invI[0])&&IsNumber(invI[1])&&IsNumber(invI[2])))
		{
            fileout<<"we choose invI=I"<<endl;
            invI[0]=1;
            invI[1]=0;
            invI[2]=1;
		}
		
	}
	I=var;
	
	

}


void braindt::two_d_var_matrix_inv( int splnumber)// compute the inverse matrix of a 2by2 matrix
{
	
	double s1,s2;
	
	vector<double> var(3);
	
	for(int i=0;i<splnumber;i++)
	{
		s1=H[i][0]-H_mean[0];
		var[0]+=s1*s1;
		s2=H[i][1]-H_mean[1];
		var[2]+=s2*s2;
		var[1]+=s1*s2;//non-diagonal entry
		
        
		
	}
	
	
	
	splnumber-=1;
	var[0]/=splnumber;
	var[2]/=splnumber;
	var[1]/=splnumber;
    
    
	double denom=var[0]*var[2]-var[1]*var[1];
	if(denom>0)
	{
		invI[0]=var[2]/denom;
		invI[2]=var[0]/denom;
		invI[1]=-var[1]/denom;
		if(!(IsFiniteNumber(invI[0])&&IsFiniteNumber(invI[1])&&IsFiniteNumber(invI[2])&&IsNumber(invI[0])&&IsNumber(invI[1])&&IsNumber(invI[2])))
		{
			
			var[0]+=TINY;
			var[2]+=TINY;
			denom=var[0]*var[2]-var[1]*var[1];
			invI[0]=var[2]/denom;
			invI[2]=var[0]/denom;
			invI[1]=-var[1]/denom;
			if(!(IsFiniteNumber(invI[0])&&IsFiniteNumber(invI[1])&&IsFiniteNumber(invI[2])&&IsNumber(invI[0])&&IsNumber(invI[1])&&IsNumber(invI[2])))
			{
				
                invI[0]=1;
                invI[1]=0;
                invI[2]=1;
                
			}
			
			
		}
		
		
		
	}
	else {
		
		var[0]+=TINY;
		var[2]+=TINY;
		denom=var[0]*var[2]-var[1]*var[1];
		invI[0]=var[2]/denom;
		invI[2]=var[0]/denom;
		invI[1]=-var[1]/denom;
		if(!(IsFiniteNumber(invI[0])&&IsFiniteNumber(invI[1])&&IsFiniteNumber(invI[2])&&IsNumber(invI[0])&&IsNumber(invI[1])&&IsNumber(invI[2])))
		{
            
            invI[0]=1;
            invI[1]=0;
            invI[2]=1;
		}
		
	}
	I=var;
	
	
    
}


void braindt::gem(ofstream &fileout,char* filename)//The Generalized EM Algorithm
{
    
    int i,j,sum,sum_cond,sweep,ell,pi;
	int t;///////
	
    
    
	fileout<<"The beginning of GEM with initial values:"<<endl;
	fileout<<"beta_hat="<<beta_hat<<endl;
	fileout<<"h_hat="<<h_hat<<endl;
    
	for(ell=0;ell<L;ell++)
	{
		fileout<<"mu1_hat["<<ell<<"]="<<mu1_hat[ell]<<endl;
		fileout<<"sigma1_sq_hat["<<ell<<"]="<<sigma1_sq_hat[ell]<<endl;
		fileout<<"pEll_hat["<<ell<<"]="<<pEll_hat[ell]<<endl;
		
		
	}
	fileout<<"End the beginning of GEM"<<endl<<endl;
    
    
    
    
    double TINY_L=TINY/L;
    
    int error_time=0;
    
    double temp_like;
    
    


	
	int seed=0;

	CRandomMersenne rnd1(seed);
	StochasticLib1  rnd2(seed);
	
	
	double a,b;//for penalized MLE (PMLE) when L>=2
	a=1;
	b=2;
	
	
	double del=1e-3;
	
	
	

	int succ_t=0;// to see the stopping rule successive time in the t iteration


	

	

    
    
	

	double exp_x[7];
	vector<double> exp_y(pN );

	
	double beta_hat_pre,h_hat_pre;//previous values for t iteration, i.e., the interations of GEM algorithm

	
	
	double d1=1;//error in t iteration of GEM
	
	vector<double> diff(2+3*L);
	
	
	
	double p_1;
	
	
	double temp2;

	
	vector<double> constant(L );
	
	
	
	vector<int> x(N );//x is the unobservable state 0 or 1
	vector<int> x_cond(N );// conditional x on y

	
	double constant_0=sqrt(2*M_PI*sigma0_sq);
	vector<double> temp1(pN);
	
	
	
	for(pi=0;pi<pN ;pi++)
	{
		
		initial[pi]=rnd1.Random()<0.5?1:0;
		temp1[pi]=constant_0*exp(pow(y[pi]-mu0,2)/(2*sigma0_sq));
	}

     
    
        vector<double> log_q2_sub(swp_r); // -phi^T*H(x)
    alam=0;
    int n_tout=1;

for(t=0;n_tout && t<max_iter;t++)
{
    
        fileout<<"The GEM iteration t="<<t<<endl;
	
		beta_hat_pre=beta_hat;
		h_hat_pre=h_hat; 
		
		vector<double> gamma_1(pN);
		
		for(i=0;i<L;i++)
		{
			constant[i]=pEll_hat[i]/sqrt(2*M_PI*sigma1_sq_hat[i]);
			
		}
		
		for(pi=0;pi<pN;pi++)
		{
			
			temp2=0;
			i=loca[pi];
			x[i]=x_cond[i]=initial[pi];
			for(j=0;j<L;j++)
			{
				temp2+=constant[j]*exp(-pow(y[pi]-mu1_hat[j],2)/(2*sigma1_sq_hat[j]));
			}
			exp_y[pi]=temp1[pi]*temp2;
		}

	for(i=0; i<7; i++)
	{
		exp_x[i]=exp(beta_hat*i+h_hat);
	}  
	
	
	H_cond_mean=H_mean0;//set to be zeros

	for(sweep=1;sweep<swp_b;sweep++)
	{
		for(pi=0; pi<pN;pi++)
		{
			i=loca[pi];
		
            
            
            sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
            sum_cond=x_cond[i+1]+x_cond[i-1]+x_cond[i+Lx]+x_cond[i-Lx]+x_cond[i+YbyX]+x_cond[i-YbyX];
			
			p_1=1-1/(1+exp_x[sum]);
			x[i]=rnd2.Bernoulli(p_1);
			p_1=1-1/(1+exp_x[sum_cond]*exp_y[pi]);
			x_cond[i]=rnd2.Bernoulli(p_1);

		}

	}	
	


	   H_mean=H_mean0;//set to be zeros

	   H=H0;
	
	for(sweep=0;sweep<swp_r;sweep++)
	{
        
        
        
        
        for(pi=0; pi<pN;pi++)
		{
            
			i=loca[pi];
            
            sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
            sum_cond=x_cond[i+1]+x_cond[i-1]+x_cond[i+Lx]+x_cond[i-Lx]+x_cond[i+YbyX]+x_cond[i-YbyX];
			
			
			
			p_1=1-1/(1+exp_x[sum]);
            
			x[i]=rnd2.Bernoulli(p_1);
			p_1=1-1/(1+exp_x[sum_cond]*exp_y[pi]);
            
			x_cond[i]=rnd2.Bernoulli(p_1);
			
		}

		for(pi=0; pi<pN;pi++)
		{

			i=loca[pi];
            
            sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
            sum_cond=x_cond[i+1]+x_cond[i-1]+x_cond[i+Lx]+x_cond[i-Lx]+x_cond[i+YbyX]+x_cond[i-YbyX];
			
			
			
		
			H_cond_mean[0]+=x_cond[i]*sum_cond;//warning:1/2 in the later, for compute twice//Don't forgot to divide swp_r at the end
			H_cond_mean[1]+=x_cond[i];
			
			
			
			H[sweep][0]+=x[i]*sum*0.5;
			H[sweep][1]+=x[i];
			
			gamma_1[pi]+=x_cond[i];//r_[i][1]
//////////////////////////////////////////////////////////////
            

        
			
		}
		
        log_q2_sub[sweep]=-H[sweep][0]*beta_hat-H[sweep][1]*h_hat;
//////////////////////////////////////////////////////////////////
		
		H_mean[0]+=H[sweep][0];//don't forgot to divide swp_r at the end
		H_mean[1]+=H[sweep][1];

		
	}
///////////////////////////////////////////////////////////////////////
    temp_like=*max_element(log_q2_sub.begin(), log_q2_sub.end());
    log_plikelihood=0;
    for(sweep=0;sweep<swp_r;sweep++)
    {
        log_plikelihood+=exp(log_q2_sub[sweep]-temp_like);
    }
    
    q2=temp_like+log(log_plikelihood);//-logZ
    //fileout<<"q2="<<q2<<endl;


    
    
    
	///////////////// U	  
	H_cond_mean[0]/=2;
    
	fileout<<"The score function U:"<<endl;
	
	for(i=0;i<2;i++)
	{
	
		H_cond_mean[i]/=swp_r;
		H_mean[i]/=swp_r;

		U[i]=H_cond_mean[i]-H_mean[i];		
                fileout<<"U["<<i<<"]="<<U[i]<<endl;
                fileout<<"H_cond_mean["<<i<<"]="<<H_cond_mean[i]<<endl;
                fileout<<"H_mean["<<i<<"]="<<H_mean[i]<<endl;
	}	
	
     
	fileout<<"The information matrix I:"<<endl;
	two_d_var_matrix_inv(fileout,swp_r);// for I update
//////////////////////////revised by 20140219
   
    
    
		for(pi=0;pi<pN;pi++)
		{
			gamma_1[pi]=gamma_1[pi]/swp_r;
			
		}
	
		double gamma_1_sum=sigmasum(gamma_1,0,pN);

		
	
    
    
		mu1_hat_pre=mu1_hat;
		sigma1_sq_hat_pre=sigma1_sq_hat;
		pEll_hat_pre=pEll_hat;			  
		
		if(L==1)
		{
			pEll_hat[0]=1;
			mu1_hat[0]=(sigmasum(gamma_1,y,0,pN)+TINY)/(gamma_1_sum+TINY);//+TINY to avoid the case gamma_1_sum=0
			sigma1_sq_hat[0]=(sigmasum(gamma_1,y,mu1_hat[0],0,pN )+TINY)/(gamma_1_sum+TINY);
		}
		else 
		{


			vector<double> f1y(pN);
			vector<double> omega_sum(L);
			vector<vector<double> > omega(L,vector<double>(pN));
			
			for(pi=0;pi<pN;pi++)
			{
				
				
				for(ell=0;ell<L;ell++)
				{
					f1y[pi]+=pEll_hat[ell]*normpdf(y[pi], mu1_hat[ell], sigma1_sq_hat[ell]); 
					
				}
				for(ell=0;ell<L;ell++)
				{
					omega[ell][pi]=gamma_1[pi]*pEll_hat[ell] \
					*normpdf(y[pi],mu1_hat[ell],sigma1_sq_hat[ell])/f1y[pi];
					omega_sum[ell]+=omega[ell][pi];
				}
			}
			

			
			
			
			for(ell=0;ell<L;ell++)
			{
			
               
                    
				mu1_hat[ell]=(sigmasum(omega[ell],y,0,pN )+TINY)/(omega_sum[ell]+TINY);// +TINY to avoid omega_sum[ell]=0
				sigma1_sq_hat[ell]=(sigmasum(omega[ell],y,mu1_hat[ell],0,pN )+2*a)/(omega_sum[ell]+2*b);//PMLE
                
              

                
                
                if(ell==(L-1))
				{
					pEll_hat[ell]=0;
					for(int ell_sub=0;ell_sub<ell;ell_sub++)
					{
						pEll_hat[ell]+=pEll_hat[ell_sub];
					}
					pEll_hat[ell]=1-pEll_hat[ell];
                    
                    
                    
                    
                    ///////if pELL_hat[L-1]<0, because of precision problem, to be like -2.22045e-16
                    if(pEll_hat[ell]<0)
                    {
                        pEll_hat[(ell-1)]+=pEll_hat[ell];
                        pEll_hat[ell]=0;
                    }
                    
                    
				}
				else {
					pEll_hat[ell]=(omega_sum[ell]+TINY_L)/(gamma_1_sum+TINY);//+TINY to avoid gamma_1_sum=0
                   // fileout<<"omega_sum["<<ell<<"]="<<omega_sum[ell]<<endl;
                   // fileout<<"gamma_1_sum="<<gamma_1_sum<<endl;
				}
                
                
		
			}
			
	
			
			
			
			
		}
	

		
/* 		
    fileout<<"before backtrack"<<endl;
    fileout<<"beta_hat="<<beta_hat<<endl;
    fileout<<"h_hat="<<h_hat<<endl;
    fileout<<"U[0]="<<U[0]<<endl;
    fileout<<"U[1]="<<U[1]<<endl;
 */
	
	     fileout<<"Go to backtracking linear seach"<<endl;
		 backtrack(fileout,rnd2);
	     fileout<<"Back to GEM from backtracking linear seach"<<endl;
//m iteration is over
	



		///////////////////////
		diff[0]=fabs(beta_hat_pre-beta_hat)/(fabs(beta_hat_pre)+del);
		diff[1]=fabs(h_hat_pre-h_hat)/(fabs(h_hat_pre)+del);
		
        fileout<<"The estimates obtained in the iteration of GEM t="<<t<<":"<<endl;
		fileout<<"beta_hat="<<beta_hat<<endl;
		fileout<<"h_hat="<<h_hat<<endl;
		
		for(ell=0;ell<L;ell++)
		{
            if(pEll_hat_pre[ell]<1e-3 && pEll_hat[ell]<1e-3)
            {
                fileout<<"pEll_hat_pre and pEll_hat are too close to 0 to be accounted into d1"<<endl;
                diff[2+ell]=0;
                diff[2+L+ell]=0;
                diff[2+2*L+ell]=0;
                
            }
            else
            {
            
			   diff[2+ell]=fabs(mu1_hat_pre[ell]-mu1_hat[ell])/(fabs(mu1_hat_pre[ell])+del );
			   diff[2+L+ell]=fabs(sigma1_sq_hat_pre[ell]-sigma1_sq_hat[ell])/(fabs(sigma1_sq_hat_pre[ell])+del );
			   diff[2+2*L+ell]=fabs(pEll_hat_pre[ell]-pEll_hat[ell])/(fabs(pEll_hat_pre[ell])+del );
                
                
                fileout<<"mu1_hat["<<ell<<"]="<<mu1_hat[ell]<<endl;
                fileout<<"sigma1_sq_hat["<<ell<<"]="<<sigma1_sq_hat[ell]<<endl;
                fileout<<"pEll_hat["<<ell<<"]="<<pEll_hat[ell]<<endl;
                
                
                
			  
            }
		}
		d1=*max_element(diff.begin(), diff.end());
		fileout<<"The scaled distance between previous and current estimates, d1="<<d1<<endl<<endl;
	
	
	   if(d1<1e-3 && alam==1 && stpover==0)
       {
           succ_t++;
       }
       else
       {
           succ_t=0;
       }
    
       if(succ_t==3)
       {
           n_tout=0;//satisfying the stopping rule for consecutive three times
       }
    
    
        
	
		
		
}// t iteration is over
    
    
    

/////////////////////////////////////////////////////////
 //   ofstream fileout;
    ofstream fileout2;
    

	fileout2.open (filename);
	
    fileout<<"Summary:"<<endl;
    fileout<<"The number of voxels, pN="<<pN<<endl;
    fileout<<"The iterations of GEM used="<<t<<endl;
    fileout<<"The indicator of satisfying the stopping rule for consecutive three times="<<1-n_tout<<endl;
    fileout<<"The scaled distance between previous and current estimates, d1="<<d1<<endl;
    fileout<<"The proportion of the regular Newton step used in the final GEM iteration,lambda="<<alam<<endl;
	
	fileout<<"The score function U:"<<endl;
    fileout<<"U[0]="<<U[0]<<endl;
    
    fileout<<"H_cond_mean[0]="<<H_cond_mean[0]<<endl;
    fileout<<"H_mean[0]="<<H_mean[0]<<endl;
    
    fileout<<"U[1]="<<U[1]<<endl;
    fileout<<"H_cond_mean[1]="<<H_cond_mean[1]<<endl;
    fileout<<"H_mean[1]="<<H_mean[1]<<endl;
	
	fileout<<"The final estimates:"<<endl;
	fileout<<"beta_hat="<<beta_hat<<endl;
	fileout<<"h_hat="<<h_hat<<endl;
    
 	for(ell=0;ell<L;ell++)
	{
		fileout<<"mu1_hat["<<ell<<"]="<<mu1_hat[ell]<<endl;
		fileout<<"sigma1_sq_hat["<<ell<<"]="<<sigma1_sq_hat[ell]<<endl;
		fileout<<"pEll_hat["<<ell<<"]="<<pEll_hat[ell]<<endl;
		
		
	}
    
	fileout2<<beta_hat<<" "<<h_hat;
    for(ell=0;ell<L;ell++)
	{
		fileout2<<" "<<mu1_hat[ell];
		fileout2<<" "<<sigma1_sq_hat[ell];
		fileout2<<" "<<pEll_hat[ell];
	}
    

    
        fileout2.close();
}


void braindt::gem(char* filename)//The Generalized EM Algorithm
{
    
    int i,j,sum,sum_cond,sweep,ell,pi;
	int t;///////
	
    
    

    
    
    double TINY_L=TINY/L;
    
    int error_time=0;
    
    double temp_like;
    
    
    
    
	
	int seed=0;
    
	CRandomMersenne rnd1(seed);
	StochasticLib1  rnd2(seed);
	
	
	double a,b;//for penalized MLE (PMLE) when L>=2
	a=1;
	b=2;
	
	
	double del=1e-3;
	
	
	
    
	int succ_t=0;// to see the stopping rule successive time in the t iteration
    

	

	
    
    
    
	
    
	double exp_x[7];
	vector<double> exp_y(pN );
    
	
	double beta_hat_pre,h_hat_pre;//for t iteration, i.e., the interations of GEM algorithm

		
	double d1=1;//error in t iteration of GEM

	vector<double> diff(2+3*L);
	
	
	
	double p_1;
	
	
	double temp2;
    
	
	vector<double> constant(L );
	
	
	
	vector<int> x(N );//x is the unobservable state 0 or 1
	vector<int> x_cond(N );// conditional x on y
    
	
	double constant_0=sqrt(2*M_PI*sigma0_sq);
	vector<double> temp1(pN);
	
	
	
	for(pi=0;pi<pN ;pi++)
	{
		
		initial[pi]=rnd1.Random()<0.5?1:0;
		temp1[pi]=constant_0*exp(pow(y[pi]-mu0,2)/(2*sigma0_sq));
	}
    
    
    
    vector<double> log_q2_sub(swp_r); // -phi^T*H(x)
    alam=0;
    int n_tout=1;
    
    for(t=0;n_tout && t<max_iter;t++)
    {
        
        
		beta_hat_pre=beta_hat;
		h_hat_pre=h_hat;
		
		vector<double> gamma_1(pN);
		
		for(i=0;i<L;i++)
		{
			constant[i]=pEll_hat[i]/sqrt(2*M_PI*sigma1_sq_hat[i]);
			
		}
		
		for(pi=0;pi<pN;pi++)
		{
			
			temp2=0;
			i=loca[pi];
			x[i]=x_cond[i]=initial[pi];
			for(j=0;j<L;j++)
			{
				temp2+=constant[j]*exp(-pow(y[pi]-mu1_hat[j],2)/(2*sigma1_sq_hat[j]));
			}
			exp_y[pi]=temp1[pi]*temp2;
		}
        
        for(i=0; i<7; i++)
        {
            exp_x[i]=exp(beta_hat*i+h_hat);
        }
        
        
        H_cond_mean=H_mean0;//set to be zeros
        
        for(sweep=1;sweep<swp_b;sweep++)
        {
            for(pi=0; pi<pN;pi++)
            {
                i=loca[pi];
                
                
                
                sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
                sum_cond=x_cond[i+1]+x_cond[i-1]+x_cond[i+Lx]+x_cond[i-Lx]+x_cond[i+YbyX]+x_cond[i-YbyX];
                
                p_1=1-1/(1+exp_x[sum]);
                x[i]=rnd2.Bernoulli(p_1);
                p_1=1-1/(1+exp_x[sum_cond]*exp_y[pi]);
                x_cond[i]=rnd2.Bernoulli(p_1);
                
            }
            
        }
        
        
        
        H_mean=H_mean0;//set to be zeros
        
        H=H0;
        
        for(sweep=0;sweep<swp_r;sweep++)
        {
            
           
            
            
            for(pi=0; pi<pN;pi++)
            {
                
                i=loca[pi];
                
                sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
                sum_cond=x_cond[i+1]+x_cond[i-1]+x_cond[i+Lx]+x_cond[i-Lx]+x_cond[i+YbyX]+x_cond[i-YbyX];
                
                
                
                p_1=1-1/(1+exp_x[sum]);
                
                x[i]=rnd2.Bernoulli(p_1);
                p_1=1-1/(1+exp_x[sum_cond]*exp_y[pi]);
                
                x_cond[i]=rnd2.Bernoulli(p_1);
                
            }
            
            for(pi=0; pi<pN;pi++)
            {
                
                i=loca[pi];
                
                sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
                sum_cond=x_cond[i+1]+x_cond[i-1]+x_cond[i+Lx]+x_cond[i-Lx]+x_cond[i+YbyX]+x_cond[i-YbyX];
                
                
                
                
                H_cond_mean[0]+=x_cond[i]*sum_cond;//warning:1/2 in the later, for compute twice//Don't forgot to divide swp_r at the end
                H_cond_mean[1]+=x_cond[i];
                
                
                
                H[sweep][0]+=x[i]*sum*0.5;
                H[sweep][1]+=x[i];
                
                gamma_1[pi]+=x_cond[i];//r_[i][1]
                //////////////////////////////////////////////////////////////
                
                
                
                
            }
            
            log_q2_sub[sweep]=-H[sweep][0]*beta_hat-H[sweep][1]*h_hat;
            //////////////////////////////////////////////////////////////////
            
            H_mean[0]+=H[sweep][0];//don't forgot to divide swp_r at the end
            H_mean[1]+=H[sweep][1];
            
            
        }
        ///////////////////////////////////////////////////////////////////////
        temp_like=*max_element(log_q2_sub.begin(), log_q2_sub.end());
        log_plikelihood=0;
        for(sweep=0;sweep<swp_r;sweep++)
        {
            log_plikelihood+=exp(log_q2_sub[sweep]-temp_like);
        }
        
        q2=temp_like+log(log_plikelihood);//-logZ

        
        
        
        
        
        ///////////////// U
        H_cond_mean[0]/=2;
        
        for(i=0;i<2;i++)
        {
            
            H_cond_mean[i]/=swp_r;
            H_mean[i]/=swp_r;
            
            U[i]=H_cond_mean[i]-H_mean[i];
            
        }
        
        
        
        two_d_var_matrix_inv( swp_r);// for I update
        //////////////////////////revised by 20140219
        
        
        
		for(pi=0;pi<pN;pi++)
		{
			gamma_1[pi]=gamma_1[pi]/swp_r;
			
		}
        
		double gamma_1_sum=sigmasum(gamma_1,0,pN);
        
		
        
        
        
		mu1_hat_pre=mu1_hat;
		sigma1_sq_hat_pre=sigma1_sq_hat;
		pEll_hat_pre=pEll_hat;
		
		if(L==1)
		{
			pEll_hat[0]=1;
			mu1_hat[0]=(sigmasum(gamma_1,y,0,pN)+TINY)/(gamma_1_sum+TINY);//+TINY to avoid the case gamma_1_sum=0
			sigma1_sq_hat[0]=(sigmasum(gamma_1,y,mu1_hat[0],0,pN )+TINY)/(gamma_1_sum+TINY);
		}
		else
		{
            
            
			vector<double> f1y(pN);
			vector<double> omega_sum(L);
			vector<vector<double> > omega(L,vector<double>(pN));
			
			for(pi=0;pi<pN;pi++)
			{
				
				
				for(ell=0;ell<L;ell++)
				{
					f1y[pi]+=pEll_hat[ell]*normpdf(y[pi], mu1_hat[ell], sigma1_sq_hat[ell]);
					
				}
				for(ell=0;ell<L;ell++)
				{
					omega[ell][pi]=gamma_1[pi]*pEll_hat[ell] \
					*normpdf(y[pi],mu1_hat[ell],sigma1_sq_hat[ell])/f1y[pi];
					omega_sum[ell]+=omega[ell][pi];
				}
			}
			
            
			
			
			
			for(ell=0;ell<L;ell++)
			{
                
                
                
				mu1_hat[ell]=(sigmasum(omega[ell],y,0,pN )+TINY)/(omega_sum[ell]+TINY);// +TINY to avoid omega_sum[ell]=0
				sigma1_sq_hat[ell]=(sigmasum(omega[ell],y,mu1_hat[ell],0,pN )+2*a)/(omega_sum[ell]+2*b);//PMLE
                
                
                
                
                
                if(ell==(L-1))
				{
					pEll_hat[ell]=0;
					for(int ell_sub=0;ell_sub<ell;ell_sub++)
					{
						pEll_hat[ell]+=pEll_hat[ell_sub];
					}
					pEll_hat[ell]=1-pEll_hat[ell];
                    
                    
                    
                    
                    ///////if pELL_hat[L-1]<0, because of precision problem, to be like -2.22045e-16
                    if(pEll_hat[ell]<0)
                    {
                        pEll_hat[(ell-1)]+=pEll_hat[ell];
                        pEll_hat[ell]=0;
                    }
                    
                    
				}
				else {
					pEll_hat[ell]=(omega_sum[ell]+TINY_L)/(gamma_1_sum+TINY);//+TINY to avoid gamma_1_sum=0
                   
				}
                
                
                
			}
			
            
			
			
			
			
		}
        
        
		
 	
        backtrack(rnd2);
        
        //m iteration is over
        
        
        
        
		///////////////////////
		diff[0]=fabs(beta_hat_pre-beta_hat)/(fabs(beta_hat_pre)+del);
		diff[1]=fabs(h_hat_pre-h_hat)/(fabs(h_hat_pre)+del);
		
        
		
		
		for(ell=0;ell<L;ell++)
		{
            if(pEll_hat_pre[ell]<1e-3 && pEll_hat[ell]<1e-3)
            {
                
                diff[2+ell]=0;
                diff[2+L+ell]=0;
                diff[2+2*L+ell]=0;
                
            }
            else
            {
                
                diff[2+ell]=fabs(mu1_hat_pre[ell]-mu1_hat[ell])/(fabs(mu1_hat_pre[ell])+del );
                diff[2+L+ell]=fabs(sigma1_sq_hat_pre[ell]-sigma1_sq_hat[ell])/(fabs(sigma1_sq_hat_pre[ell])+del );
                diff[2+2*L+ell]=fabs(pEll_hat_pre[ell]-pEll_hat[ell])/(fabs(pEll_hat_pre[ell])+del );
                
                
                
                
                
            }
		}
		d1=*max_element(diff.begin(), diff.end());
	
        
        
        if(d1<1e-3 && alam==1 && stpover==0)
        {
            succ_t++;
        }
        else
        {
            succ_t=0;
        }
        
        if(succ_t==3)
        {
            n_tout=0;//satisfying the stopping rule for consecutive three times 
        }
        
        
        
        
		
		
    }// t iteration is over
    
    
    
    
    /////////////////////////////////////////////////////////
    //   ofstream fileout2;
    ofstream fileout2;
    
    
	fileout2.open (filename);
	
    
    
	fileout2<<beta_hat<<" "<<h_hat;
    for(ell=0;ell<L;ell++)
	{
		fileout2<<" "<<mu1_hat[ell];
		fileout2<<" "<<sigma1_sq_hat[ell];
		fileout2<<" "<<pEll_hat[ell];
	}
    
    
    
    fileout2.close();
}


void braindt::func(ofstream &fileout,StochasticLib1 &rnd2)//update U and q2 in the Armijo condition
{
    vector<double> log_q2_sub(swp_r);// -phi^T*H(x)
   
    double temp_like;
    H_mean=H_mean0;
    H=H0;
	int pi,i,sweep,sum;
	double p_1;
	vector<int> x(N );
	for(pi=0;pi<pN;pi++)
	{
		i=loca[pi];
		x[i]=initial[pi];
	}
	
	
	double exp_x[7];
	for(i=0; i<7; i++)
	{
		exp_x[i]=exp(beta_hat*i+h_hat);
	}  
	for(sweep=1;sweep<swp_b;sweep++)
	{
		for(pi=0; pi<pN;pi++)
		{
			
			
			
			i=loca[pi];
            
            sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
 
			
			p_1=1-1/(1+exp_x[sum]);
			
			x[i]=rnd2.Bernoulli(p_1);
			
			
		}
		
	}	
	

	
	for(sweep=0;sweep<swp_r;sweep++)
	{
      
        for(pi=0; pi<pN;pi++)
		{
			
			
			
			i=loca[pi];
            
			sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
			p_1=1-1/(1+exp_x[sum]);
			
			x[i]=rnd2.Bernoulli(p_1);
			
			
		}
		
		
		for(pi=0;pi<pN;pi++)
		{
			
			
			
			i=loca[pi];
            
			sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
			
			
			
			H[sweep][0]+=x[i]*sum*0.5;
			H[sweep][1]+=x[i];
            //////////////////////////////////////////////////////////////
           
            
			
		}
		log_q2_sub[sweep]=-H[sweep][0]*beta_hat-H[sweep][1]*h_hat;
       
        //////////////////////////////////////////////////////////////////
			
		
		H_mean[0]+=H[sweep][0];//don't forgot to divide  swp_r at the end
		H_mean[1]+=H[sweep][1];
		
		
	}
	///////////////////////////////

    
    
    
    temp_like=*max_element(log_q2_sub.begin(), log_q2_sub.end());
    log_plikelihood=0;
    for(sweep=0;sweep<swp_r;sweep++)
    {
        log_plikelihood+=exp(log_q2_sub[sweep]-temp_like);
    }
    
    q2=temp_like+log(log_plikelihood);//-logZ
    //fileout<<"q2="<<q2<<endl;
    

	fileout<<"The score function U:"<<endl;

	for(i=0;i<2;i++)
	{
		H_mean[i]/=swp_r;
		
		U[i]=H_cond_mean[i]-H_mean[i];
		
		fileout<<"U["<<i<<"]="<<U[i]<<endl;
		fileout<<"H_cond_mean["<<i<<"]="<<H_cond_mean[i]<<endl;
		fileout<<"H_mean["<<i<<"]="<<H_mean[i]<<endl;
		
	}
	
	
	
	
}
void braindt::func(StochasticLib1 &rnd2)//update U and q2 in the Armijo condition
{
    vector<double> log_q2_sub(swp_r);// -phi^T*H(x)
   
    double temp_like;
    H_mean=H_mean0;
    H=H0;
	int pi,i,sweep,sum;
	double p_1;
	vector<int> x(N );
	for(pi=0;pi<pN;pi++)
	{
		i=loca[pi];
		x[i]=initial[pi];
	}
	
	
	double exp_x[7];
	for(i=0; i<7; i++)
	{
		exp_x[i]=exp(beta_hat*i+h_hat);
	}
	for(sweep=1;sweep<swp_b;sweep++)
	{
		for(pi=0; pi<pN;pi++)
		{
			
			
			
			i=loca[pi];
            
            sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
            
			
			p_1=1-1/(1+exp_x[sum]);
			
			x[i]=rnd2.Bernoulli(p_1);
			
			
		}
		
	}
	
    
	
	for(sweep=0;sweep<swp_r;sweep++)
	{
      
        for(pi=0; pi<pN;pi++)
		{
			
			
			
			i=loca[pi];
            
			sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
			p_1=1-1/(1+exp_x[sum]);
			
			x[i]=rnd2.Bernoulli(p_1);
			
			
		}
		
		
		for(pi=0;pi<pN;pi++)
		{
			
			
			
			i=loca[pi];
            
			sum=x[i+1]+x[i-1]+x[i+Lx]+x[i-Lx]+x[i+YbyX]+x[i-YbyX];
			
			
			
			H[sweep][0]+=x[i]*sum*0.5;
			H[sweep][1]+=x[i];
            //////////////////////////////////////////////////////////////
            
            
			
		}
		log_q2_sub[sweep]=-H[sweep][0]*beta_hat-H[sweep][1]*h_hat;
        
        //////////////////////////////////////////////////////////////////
        
		
		H_mean[0]+=H[sweep][0];//don't forgot to divide  swp_r at the end
		H_mean[1]+=H[sweep][1];
		
		
	}
	///////////////////////////////
    
    
    
    
    temp_like=*max_element(log_q2_sub.begin(), log_q2_sub.end());
    log_plikelihood=0;
    for(sweep=0;sweep<swp_r;sweep++)
    {
        log_plikelihood+=exp(log_q2_sub[sweep]-temp_like);
    }
    
    q2=temp_like+log(log_plikelihood);//-logZ
    
    
    

	
	
}


void braindt::backtrack(ofstream &fileout,StochasticLib1 &rnd2)//The backtracking line search method
{//see P479 code of the book Numerical Recipe, 3rd
    stpover=0;
	q2_old=q2;
	
    double del=1e-3;
    
	beta_hat_old=beta_hat;
	h_hat_old=h_hat;
	


	const double ALF=1e-4;//alpha in the Armijo condition
	const double tolx=1e-4;//tolerance for error in m iteration of backtracking 
	const double stpmax=5;// the maximum length of the newton step
	
	
	double alamin;//the minimum lambda in the Armijo condition
	
	
	double temp,test;
	

	
    delta[0]=invI[0]*U[0]+invI[1]*U[1];
	delta[1]=invI[1]*U[0]+invI[2]*U[1];
	
	double sum=delta[0]*delta[0]+delta[1]*delta[1];//delta=p in the book, is the regular newton step
	sum=sqrt(sum);
	if(sum>stpmax)//Scale if attempted step is too big.
	{
        stpover=1;
		delta[0]*=stpmax/sum;
		delta[1]*=stpmax/sum;
        //fileout<<"in backtrack firt delta1="<<delta[0]<<endl;
        //fileout<<"in backtrack firt delta2="<<delta[1]<<endl;
        fileout<<"Over the max step in backtracking,the step is thus scaled"<<endl;
	}
    
    
	double slope=U[0]*delta[0]+U[1]*delta[1];
	//fileout<<"U*invI*U="<<slope<<endl;

	
	temp=fabs(delta[0])/(fabs(beta_hat_old)+del);
	test=fabs(delta[1])/(fabs(h_hat_old)+del);
	test=test>temp?test:temp;
	alamin=tolx/test;
	alam=1;
	int m=0;//
	
	
	fileout<<"The backtracking iteration m=0"<<endl;
	fileout<<"lambda="<<alam<<endl;
	
	for(;;)
	{
		if(alam<alamin)
		{
           
            fileout<<"Converge at the case that lambda less than the minimum limit"<<endl;
			beta_hat=beta_hat_old;
			h_hat=h_hat_old;
            
			break;
		}
		else {
			
			beta_hat=beta_hat_old+alam*delta[0];
			h_hat=h_hat_old+alam*delta[1];
			
			
			fileout<<"beta_hat="<<beta_hat<<endl;
			fileout<<"h_hat="<<h_hat<<endl;
			
            
			
			func(fileout,rnd2);//update q2
            

           // fileout<<"q2_old="<<q2_old<<endl;
           // fileout<<"q2="<<q2<<endl;
            delta_q2=(beta_hat-beta_hat_old)*H_cond_mean[0]+(h_hat-h_hat_old)*H_cond_mean[1]+q2-q2_old;//q2-q2old
            fileout<<"delta_Q2="<<delta_q2<<endl;
            
			
			if(delta_q2<(ALF*alam*slope))
			{
				m++;
				fileout<<"The backtracking iteration m="<<m<<endl;
                alam*=0.5;
				fileout<<"lambda="<<alam<<endl;
			}
			else{
				
				fileout<<"End backtracking for the Q2 increase"<<endl;
				break;
			}
			
		}

        
        
	}
}


void braindt::backtrack(StochasticLib1 &rnd2)//The backtracking line search method
{//see P479 code of the book Numerical Recipe, 3rd
    stpover=0;
	q2_old=q2;
	
    double del=1e-3;
    
	beta_hat_old=beta_hat;
	h_hat_old=h_hat;
	

    
	const double ALF=1e-4;//alpha in the Armijo condition
	const double tolx=1e-4;//tolerance for error in m iteration of backtracking
	const double stpmax=5;// the maximum length of the newton step
	
	
	double alamin;//the minimum lambda in the Armijo condition
	
	
	double temp,test;
	
	
	
    delta[0]=invI[0]*U[0]+invI[1]*U[1];
	delta[1]=invI[1]*U[0]+invI[2]*U[1];
	
	double sum=delta[0]*delta[0]+delta[1]*delta[1];//delta=p in the book, is the regular newton step
	sum=sqrt(sum);
	if(sum>stpmax)//Scale if attempted step is too big.
	{
        stpover=1;
		delta[0]*=stpmax/sum;
		delta[1]*=stpmax/sum;
    
	}
    
    
	double slope=U[0]*delta[0]+U[1]*delta[1];
	
	
	
	temp=fabs(delta[0])/(fabs(beta_hat_old)+del);
	test=fabs(delta[1])/(fabs(h_hat_old)+del);
	test=test>temp?test:temp;
	alamin=tolx/test;
	alam=1;

	for(;;)
	{
		if(alam<alamin)
		{
            
            
			beta_hat=beta_hat_old;
			h_hat=h_hat_old;
            
			break;
		}
		else {
			beta_hat=beta_hat_old+alam*delta[0];
			h_hat=h_hat_old+alam*delta[1];
			
			
			
			
            
			
			func(rnd2);//update q2
            
            
           
            delta_q2=(beta_hat-beta_hat_old)*H_cond_mean[0]+(h_hat-h_hat_old)*H_cond_mean[1]+q2-q2_old;//q2-q2old
            
            
			
			if(delta_q2<(ALF*alam*slope))
			{

				
                
                alam*=0.5;
			}
			else{
				
				
				break;
			}
			
		}
        
        
        
	}
}



////////////////computing LIS
void braindt::computeLIS(char* filename)
{
    
    int seed=0;
	CRandomMersenne rnd1(seed);
	StochasticLib1  rnd2(seed);
	
    
	int i,j,k,nn,sum,sum_cond,sweep,ell,pi;
	int m;///////
	
	
    
	double exp_x[7];
	vector<double> exp_y(pN );
	
	
	
	double p_1;
	
	
	double temp,temp2;
    
	
	vector<double> constant(L );
	
	vector<int> x_cond(N );// conditional x
    
	
	double constant_0=sqrt(2*M_PI*sigma0_sq);
	vector<double> temp1(pN);
	
	
	
	for(pi=0;pi<pN ;pi++)
	{
		temp1[pi]=constant_0*exp(pow(y[pi]-mu0,2)/(2*sigma0_sq));
	}
    
    
	for(i=0; i<7; i++)
	{
		exp_x[i]=exp(beta_hat*i+h_hat);
	}
    
    
    
	for(i=0;i<L;i++)
	{
		constant[i]=pEll_hat[i]/sqrt(2*M_PI*sigma1_sq_hat[i]);
		
	}
	for(pi=0;pi<pN;pi++)
	{
		i=loca[pi];
		x_cond[i]=initial[pi];//
		temp2=0;
		for(j=0;j<L;j++)
		{
			temp2+=constant[j]*exp(-pow(y[pi]-mu1_hat[j],2)/(2*sigma1_sq_hat[j]));
		}
		exp_y[pi]=temp1[pi]*temp2;
		
	}
	
    
	for(int sweep=1;sweep<swp_b;sweep++)
	{
        
		for(pi=0; pi<pN;pi++)
		{
			i=loca[pi];
            
            
            sum_cond=x_cond[i+1]+x_cond[i-1]+x_cond[i+Lx]+x_cond[i-Lx]+x_cond[i+YbyX]+x_cond[i-YbyX];
            
			p_1=1-1/(1+exp_x[sum_cond]*exp_y[pi]);
			x_cond[i]=rnd2.Bernoulli(p_1);
		}
		
		
		
        
		
		
		
		
        
		
	}
	
	for(int sweep=0;sweep<swp_lis;sweep++)
	{
        
		for(pi=0; pi<pN;pi++)
		{
			i=loca[pi];
            
            sum_cond=x_cond[i+1]+x_cond[i-1]+x_cond[i+Lx]+x_cond[i-Lx]+x_cond[i+YbyX]+x_cond[i-YbyX];
			
			
            
			p_1=1-1/(1+exp_x[sum_cond]*exp_y[pi]);
			x_cond[i]=rnd2.Bernoulli(p_1);
			
			
            LIS[pi]+=x_cond[i];
            
            
			
		}
        
	}
    
    
    ofstream file;
    
    file.open (filename);
    
    for(pi=0;pi<pN ;pi++)
    {
        LIS[pi]=1-LIS[pi]/swp_lis;
        file<<LIS[pi]<<" ";
    }
	
    file.close();
    
    
}







