#include<cstdio>
#include<cstdlib>

#include<iostream>
#include <fstream>
#include<vector>
#include"braindt.h"

#include<ctime>
#include<string>
#include<cstring>
#include<mex.h>

using namespace std;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
    clock_t start;
    if(nrhs==7)
    {
        start=clock();
    }
    /* Check for proper number of arguments. */
    if(nrhs!=6 && nrhs!=7)
    {
        mexErrMsgTxt("Six or seven inputs are required.");
    //7th input argument, the file of outputs during the algorithm process, is optional
    }
    
    //The first input 
    int Get_N0= mxGetN(prhs[0]);
    if(mxGetM(prhs[0])!=1 || !(Get_N0==4 || Get_N0==8))
       mexErrMsgTxt("Input argument 1 must be a row vector of size 4 or 8.");
    
  
    double *input1=mxGetPr(prhs[0]);
    for(int i=0;i<Get_N0;i++)
    {
        if((input1[i]-floor(input1[i]))>0 || input1[i]<1)
            mexErrMsgTxt("Input argument 1 must be a row vector with positive-integer elements.");
    }
    
    int LX=(int)input1[0];//dimension of the lattice on x-axis
    int LY=(int)input1[1];//dimension of the lattice on y-axis
    int LZ=(int)input1[2];//dimension of the lattice on z-axis
    int Num=LX*LY*LZ;//total number of voxels of the lattice on the image grid
    
    int L_hat=(int)input1[3];//the number of components in the normal mixture of non-null distribution
    
    int sweep_b=1000;//iterations in the burn-in period of gibbs sampler
    int sweep_r=5000;//the number of gibbs-sampler samples
    int sweep_lis=1e6;//the gibbs sample-size for computing LIS
    int iter_max=5000;//the maximum limit of iterations in the generalized EM algorithm
    
    if(Get_N0==8)
    {
        sweep_b=(int)input1[4];
        sweep_r=(int)input1[5];
        sweep_lis=(int)input1[6];
        iter_max=(int)input1[7];
    }
    
    
    //The second input
    int est_size=2+3*L_hat;//The number of parameters to be estimated
    if(mxGetM(prhs[1])!=1 || mxGetN(prhs[1])!=est_size)
       mexErrMsgTxt("Input argument 2 must be a row vector of size 2+3*L, where L is the number of components in the normal mixture of non-null distribution.");
    
    
    double *input2=mxGetPr(prhs[1]);
    vector<double> est(input2,input2+est_size);//initial values for parameters
  
    
    //The 3rd to 6th/7th inputs
    char *file[5];// the directory of file,
    /*file[0]: observed data (z-value) on the whole image grid including the brain part and the non-brain part
     *file[1]: location of the brain region of interest on the LX*LY*LZ image grid
     *file[2]: output of the result: parameter estimates
     *file[3]: output of LIS
     *file[4]: outputs during the algorithm process, this file is optional
     */
    int filelen[5];//the length of directory
    int status[5];
    for(int i=2;i<nrhs;i++)
    {
        // Input must be a string. 
        if (mxIsChar(prhs[i]) != 1)
           mexErrMsgTxt("Input arguments from #3 must be a string.");
        // Input must be a row vector. 
        if (mxGetM(prhs[i]) != 1)
           mexErrMsgTxt("Input arguments from #3 must be a row vector.");
        // Get the length of the input string.
        int i_2=i-2;
        filelen[i_2] = (mxGetM(prhs[i]) * mxGetN(prhs[i])) + 1;
        /* Allocate memory for input and output strings. */
        file[i_2] =(char *)mxCalloc(filelen[i_2], sizeof(char));
        //Copy the string data from prhs[i] into a C++ string *file
        status[i_2] = mxGetString(prhs[i], file[i_2], filelen[i_2]);
        if (status[i_2] != 0) 
          mexWarnMsgTxt("Not enough space. String is truncated.");
        
    }
    
    
    
    vector<int> Loca;//Location of the ROI
    ifstream ifs(file[1]);
	double temp_loca;
	while(ifs>>temp_loca)
		Loca.push_back(int(temp_loca)-1);// The entry location of C++ array starts from 0, but vectorized image lattice location starts from 1
	int PN=Loca.size();// the number of voxels in the ROI

    
    
    
    
    braindt B(LX, LY, LZ,Num,file[0],Loca,PN,est,L_hat,sweep_b,sweep_r,sweep_lis,iter_max);
   
    if(nrhs==7)
    {
        ofstream fileout;
        fileout.open (file[4]);
        
        B.gem(fileout,file[2]);
        B.computeLIS(file[3]);
        clock_t finish = clock();
	    double duration = (double)(finish-start)/CLOCKS_PER_SEC;
        fileout<<"time used="<<duration<<endl;
        fileout.close();
    }
    else
    {
        B.gem(file[2]);
        B.computeLIS(file[3]);
    }
 
  
}