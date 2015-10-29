%Assume the step (1) in the section "Steps to implement the HMRF-based PLIS
%procedure" of "README.pdf" was done.
mex_setup();
input_info=[160,160,96,0.1];
reglocafile_56='loca_56.txt';
pvaluefile='pvalue_1vs2.txt';
output_signal_56='signals_BH_56.txt';
result=fdrBH95(input_info,reglocafile_56,pvaluefile,output_signal_56)

reglocafile_40='loca_40.txt';
output_signal_40='signals_BH_40.txt';
result=fdrBH95(input_info,reglocafile_40,pvaluefile,output_signal_40)

zvaluefile='zvalue_1vs2.txt';
L=2;

initials_56=initialvalue(output_signal_56,zvaluefile,reglocafile_56,L)
initials_40=initialvalue(output_signal_40,zvaluefile,reglocafile_40,L)


input_info=[160,160,96,2,1000,5000,1e6,5000];

output_estimate_56='estimate_result_56.txt';
output_LIS_56='LIS_56.txt';
output_process_56='process_output_56.txt';

output_estimate_40='estimate_result_40.txt';
output_LIS_40='LIS_40.txt';
output_process_40='process_output_40.txt';

computelis(input_info,initials_56,zvaluefile,reglocafile_56,output_estimate_56,output_LIS_56,output_process_56)
%It takes about 2.5 minutes run
computelis(input_info,initials_40,zvaluefile,reglocafile_40,output_estimate_40,output_LIS_40,output_process_40)
%It takes about 6.5 minutes to run
%In this example, you can run fdrplis without running computelis ahead, because all files
%needed to run fdrplis are areadly included in the folder. 

input_info=[160,160,96,1e-3];
reg_no=[40,56];
reglocafile='loca_';
LISfile='LIS_';
output_signal='signals_plis_40&56.txt';
output_LIS='LIS_40&56.txt';

result=fdrplis(input_info,reg_no,reglocafile,LISfile,output_signal,output_LIS)
