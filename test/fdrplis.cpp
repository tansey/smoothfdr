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
double sigmasum(vector<double> &p, int begin, int end);
void fdr_plis(double &q,int pN,vector<double> &LIS,vector<int> &signal_lis,int &R,double &threshold);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{

    if(nrhs!=6)
    {
        mexErrMsgTxt("Six inputs are required.");
    }
    
    //The first input 
    int Get_N1= mxGetN(prhs[0]);
    if(mxGetM(prhs[0])!=1 || Get_N1!=4)
       mexErrMsgTxt("Input argument 1 must be a row vector of size 4.");
    
  
    double *input1=mxGetPr(prhs[0]);
    for(int i=0;i<3;i++)
    {
        if((input1[i]-(double((int)input1[i])))>0 || input1[i]<1)
            mexErrMsgTxt("The first three elements of input argument 1 must be positive integers.");
    }
    
    if(!(input1[3]<1 && input1[3]>0))
        mexErrMsgTxt("The last element of input argument 1 must be greater than 0 and less than 1.");
    
    int LX=(int)input1[0];//length of the lattice on x-axis
    int LY=(int)input1[1];//length of the lattice on y-axis
    int LZ=(int)input1[2];//length of the lattice on z-axis
    double q=input1[3];// control fdr at level q
    
    int Num=LX*LY*LZ;//total voxels of the lattice on the image grid
    
    //The second input is the # of each brain region of interest (ROI)
    double *input2=mxGetPr(prhs[1]);
    int Get_N2= mxGetN(prhs[1]);
    for(int i=0;i<Get_N2;i++)
    {
        if((input2[i]-(double((int)input2[i])))>0 || input2[i]<0)
            mexErrMsgTxt("Input argument 2 must be a row vector with nonnegative-integer elements.");
    }
    
    
  
 //////////////////////////////////
    //The 3rd to 6th inputs
    char *file[4];// the directory of file,
    /*file[0]: location file for each ROI on the LX*LY*LZ image grid
     *file[1]: LIS file for each ROI
     *file[2]: output file of signals for all ROIs on the LX*LY*LZ image grid, where 1 indicates signal and 0 indicates nonsignal
     *file[3]: output file of LIS values for all ROIs on the LX*LY*LZ image grid
     */
    int filelen[4];//the length of directory
    int status[4];

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
    
    
    
    
    vector<int> Loca;//Location of all ROIs
    vector<double> roi_LIS;// LIS values of all ROIs
    
    string file_loca(file[0]);
    string file_lis(file[1]);
    string no_str;
    
    string txt=".txt";

    string filename;
    int no=0;// the number of each ROI
    
    double temp;
    
    for(int i=0;i<Get_N2;i++)
    {
        no=(int)input2[i];
        ostringstream convert;
        convert<<no;
        no_str=convert.str();
        filename=file_loca+no_str+txt;
        ifstream ifs(filename.c_str());
        while(ifs>>temp)
            Loca.push_back(int(temp)-1); // The entry location of C++ array starts from 0, but vectorized image lattice location starts from 1
        
        
        filename=file_lis+no_str+txt;
        ifstream ifs2(filename.c_str());
        while(ifs2>>temp)
            roi_LIS.push_back(temp);
    
    }
    
           
   int roi_pN=Loca.size();//the total number of voxels in all ROIs
    
    
    vector<int> roi_signal(roi_pN);//signals of all ROIs
    
    
    
    double threshold;//threshold in the plis-fdr procedure
	int R;//the number of signals (rejections)

    fdr_plis(q,roi_pN,roi_LIS,roi_signal,R,threshold);

    vector<int>  global_signal(Num);//output signals of all ROIs on the LX*LY*LZ image grid
    
    vector<double>  global_LIS(Num);//output LIS values of all ROIs on the LX*LY*LZ image grid
    
    

    
    int i;
    for(int pi=0;pi<roi_pN;pi++)
    {
        i=Loca[pi];
        global_signal[i]=roi_signal[pi];
        global_LIS[i]=roi_LIS[pi];
    }
    
    
    ofstream signal_file;
    ofstream lis_file;
    
    signal_file.open(file[2]);
    lis_file.open(file[3]);
    
    
    for(i=0;i<Num;i++)
    {
        signal_file<<global_signal[i]<<" ";
        lis_file<<global_LIS[i]<<" ";
    }
    
    signal_file.close();
    lis_file.close();
    
    plhs[0]=mxCreateDoubleMatrix(1,3,mxREAL);
    double *out;
    
    out=mxGetPr(plhs[0]);
    
    out[0]=R;
    out[1]=roi_pN;
    out[2]=threshold;

    
    
}
double sigmasum(vector<double> &p, int begin, int end)
{
	double sum=0;
	for(int i=begin; i<end; i++)
	{
		sum+=p[i];
	}
	return sum;
}
void fdr_plis(double &q,int pN,vector<double> &LIS,vector<int> &signal_lis,int &R,double &threshold)
{
	vector<double> sort_lis(LIS);
	sort(sort_lis.begin(), sort_lis.end());
	
	if(sort_lis[0]<=q)
	{
		for(R=1; R<=pN && sigmasum(sort_lis, 0, R)<=R*q; R++);

		R=R-1;
		
		threshold=sort_lis[R-1];
        
		
		int time=0;// for the LIS ties problem
		for(int i=0; i<pN && time<R; i++)
		{
			if(LIS[i]<=threshold)
			{
				signal_lis[i]=1;
                time++;
			}
          
		}
	}
    else
    {
        R=0;
        threshold=q;
    }
}