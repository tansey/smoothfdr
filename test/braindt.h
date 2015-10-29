#ifndef briandtH 

#define braindtH

#include<iostream>
#include <fstream>
#include<vector>
#include"randomc.h"
#include"stocc.h"

#include<cstring>
using namespace std;
class braindt{
public:
	braindt(int &LX, int &LY, int &LZ,int &Num,char *argv ,vector<int> &Loca,int &PN,vector<double>& est,int &L_hat,int sweep_b, int sweep_r,int sweep_lis,int iter_max);//constructor function of class braindt
	
	double normpdf(double &x, double &mu, double &sigma_sq);// The probability density function of the normal distribution with mean=mu, variance=sigma_sq
    
	bool IsFiniteNumber(double &x);// Judge whether x is a finite numebr
	bool IsNumber(double &x);//Judge whether x is a numebr
    
    
	double sigmasum(vector<double> &p, int begin, int end);//sum of elements of vector p
	double sigmasum(vector<double> &p1, vector<double> &p2, int begin, int end);//sum of elements of the entrywise-product of vectors p1 and p2
	double sigmasum(vector<double> &p1, vector<double> &p2, double p2minus, int begin, int end);//sum of elements of the entrywise-product of vectors p1 and p3 squared, where p3=p2-p2minus
	
    void two_d_var_matrix_inv(ofstream &fileout,int splnumber);//compute the inverse matrix of a 2 by 2 matrix
    void two_d_var_matrix_inv(int splnumber);

	void gem(ofstream &fileout,char* filename);//The Generalized EM Algorithm
    void gem(char* filename);

	void backtrack(ofstream &fileout, StochasticLib1 &rnd2);//The backtracking line search method
    void backtrack(StochasticLib1 &rnd2);
    
    void func(ofstream &fileout,StochasticLib1 &rnd2);//update U and q2 in the Armijo condition
    void func(StochasticLib1 &rnd2);
    
    
    void computeLIS(char* filename);//computing LIS

private:
	static const double TINY;//a very small number to avoid the zero denominator of a fraction

	int Lx,Ly,Lz,N,YbyX;//N=Lx*Ly*Lz is the size of the image grid, and YbyX=Ly*Lx

	vector<double> y;//observed data
    
	vector<int>::iterator loca;// the location of the brain region of interest (ROI) on the Lx*Ly*Lz image grid
    
	int pN;//the number of voxels in the ROI, i.e., the size of loca
    
///////////////////////////////////////////
    
    
	int L;// the number of components in the normal mixture of non-null distribtion 
	
    //parameters in the null distribution; we assume the null is standard normal.
    static const double mu0;//
	static const double sigma0_sq;//
    
    //parameters in the non-null distribution
	vector<double> mu1_hat;
	vector<double> sigma1_sq_hat;
	vector<double> pEll_hat;//the weight for each component
    
    //parameters in the Ising model
	double beta_hat;//parameter for sum of x_s*x_t,where x is the unobservable state
	double h_hat;//parameter for sum of x_s

///////////////////////////////////////////
	
	vector<double> U;// the score function	
	vector<double> I;// the information matrix 
    vector<double> invI;// the inverse information matrix
	
	int swp_b, swp_r,max_iter;//swp_b is the number of burn-in iterations, swp_r is samples from the Gibbs sampler, max_iter is the maximum limit of iterations in GEM.
    

	vector<double> delta;//the regular Newton step
    
	double beta_hat_old, h_hat_old;//previous values for m iteration, i.e., the interations in backtracking method
    
	vector<int> initial;//the initial random field for the gibbs sampler 

	
	vector<double> H_cond_mean;//mean H conditional on observed data
	
	vector<double> H_mean0;//Used to reset H_cond_mean or H_mean to be zeros
	vector<vector<double> > H0;//Used to reset H to be zeros
	vector<double> H_mean;// mean H
	vector<vector<double> > H;//H
	

	
////////////////////
    double log_plikelihood;// sum of exp(-phi^T*H(sample[i])), in the difference in Q_2

    



    //previous values for t iteration, i.e., the interations in GEM
    vector<double> mu1_hat_pre;
	vector<double> sigma1_sq_hat_pre;
	vector<double> pEll_hat_pre;
    
    
    double alam;// the lambda in the Armijo condition
    int stpover;//indicate whether over the max step

double q2,q2_old, delta_q2;// q_2 is log(sum of exp(-phi^T*H(sample[i]))), in the in the difference in Q_2 
     
////////////////////////////////////
    vector<double> LIS;
    int swp_lis;//  the gibbs sample-size for computing LIS
};

#endif




