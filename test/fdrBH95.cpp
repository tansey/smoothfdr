
#include<cstdio>
#include<cstdlib>
#include <algorithm>

#include<iostream>
#include<fstream>
#include<vector>


#include<string>
#include<cstring>
#include<sstream>//for convert int to string

#include<mex.h>

using namespace std;
void fdr_pvalue(double &q,int pN,vector<double> &p_value,vector<int> &signal_pvalue,int &R,double &threshold);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
	int i;

    if(nrhs!=4)
    {
        mexErrMsgTxt("Four inputs are required.");
    }
    
    //The first input 
    int Get_N1= mxGetN(prhs[0]);
    if(mxGetM(prhs[0])!=1 || Get_N1!=4)
       mexErrMsgTxt("Input argument 1 must be a row vector of size 4.");
    
  
    double *input1=mxGetPr(prhs[0]);
    for(i=0;i<3;i++)
    {
        if((input1[i]-(double((int)input1[i])))>0 || input1[i]<1)
            mexErrMsgTxt("The first three elements of input argument 1 must be positive integers.");
    }
    
    if(!(input1[3]<1 && input1[3]>0))
        mexErrMsgTxt("The last element of input argument 1 must be greater than 0 and less than 1.");
    
    int LX=(int)input1[0];//dimension of the lattice on x-axis
    int LY=(int)input1[1];//dimension of the lattice on y-axis
    int LZ=(int)input1[2];//dimension of the lattice on z-axis
    double q=input1[3];// control fdr at level q
    
    int Num=LX*LY*LZ;//total voxels of the lattice on the image grid
    
    
  
 //////////////////////////////////
    //The 2nd to 4th inputs
    char *file[3];// the directory of file,
    /*file[0]: location file for the brain region of interest (ROI) on the LX*LY*LZ image grid
     *file[1]: observed data (p-value) on the whole image grid including the brain part and the non-brain part
     *file[2]: output file of signals for the ROI
     */
    int filelen[3];//the length of directory
    int status[3];

    for(i=1;i<nrhs;i++)
    {
        // Input must be a string.
        if (mxIsChar(prhs[i]) != 1)
            mexErrMsgTxt("Input arguments from #3 must be a string.");
        // Input must be a row vector.
        if (mxGetM(prhs[i]) != 1)
            mexErrMsgTxt("Input arguments from #3 must be a row vector.");
        // Get the length of the input string.
        int i_1=i-1;
        filelen[i_1] = (mxGetM(prhs[i]) * mxGetN(prhs[i])) + 1;
        /* Allocate memory for input and output strings. */
        file[i_1] =(char *)mxCalloc(filelen[i_1], sizeof(char));
        //Copy the string data from prhs[i] into a C++ string *file
        status[i_1] = mxGetString(prhs[i], file[i_1], filelen[i_1]);
        if (status[i_1] != 0)
            mexWarnMsgTxt("Not enough space. String is truncated.");
        
    }
    
    
    
    
    vector<int> Loca;//Location of the ROI
    
    ifstream ifs_loca(file[0]);
	double temp_loca;
	while(ifs_loca>>temp_loca)
		Loca.push_back(int(temp_loca)-1);// The entry location of C++ array starts from 0, but vectorized image lattice location starts from 1
	int PN=Loca.size();// the number of voxels in the ROI
	
    
	
	vector<double> pvalue_all;//observed data (p-value) on the whole image grid including the brain part and the non-brain part
	ifstream ifs_pvalue(file[1]);//
	double temp_pvalue;
	while(ifs_pvalue>>temp_pvalue)
		pvalue_all.push_back(temp_pvalue);// read in y_ori
    
	vector<double> pvalue(PN);//the p-values of the ROI
	
    for(int pi=0;pi<PN;pi++)//read in just the p-value of the ROI
	{
		i=Loca[pi];
		pvalue[pi]=pvalue_all[i];
	}
	
	
	
	
	vector<int> signals(PN);//The signals of the ROI,1=signal, 0=non-signal
	
	double threshold;//threshold in the BH procedure
	int R;//the number of signals (rejections)
	
	fdr_pvalue(q,PN,pvalue,signals,R,threshold);
	
    //output the signal file
    ofstream signal_file;    
    signal_file.open(file[2]);
    
    
    for(i=0;i<PN;i++)
    {
        signal_file<<signals[i]<<" ";
    }
    
    signal_file.close();

    
    plhs[0]=mxCreateDoubleMatrix(1,3,mxREAL);
    double *out;
    
    out=mxGetPr(plhs[0]);
    
    out[0]=R;
    out[1]=PN;
    out[2]=threshold;

    
    
}
void fdr_pvalue(double &q,int pN,vector<double> &p_value,vector<int> &signal_pvalue,int &R,double &threshold)
{    
	vector<double> sort_pvalue(p_value);
	sort(sort_pvalue.begin(),sort_pvalue.end());
	
	for(R=pN; R>0 && sort_pvalue[R-1]*pN>R*q; R--);
	
	
	if(R!=0)
	{
		
		threshold=sort_pvalue[R-1];
		
		
		int time=0;// for the tie pvalue problem
		for(int j=0; j<pN && time<R; j++)
		{
			
			if(p_value[j]<=threshold)
			{
				signal_pvalue[j]=1;
                time++;
				
			}
		}		
	}
    else
    {
        R=0;
        threshold=q/pN;
    }

    
}
