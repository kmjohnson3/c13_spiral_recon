/* 

   Recon Code for Cartesian and Non-Cartesian Data
   (in process of reorganizing for template + 4D recons)


 */

#include "ArrayTemplates.cpp"
#include "tictoc.hpp"
#include <omp.h>
#include <armadillo>
#include <cmath>

using namespace std;
using namespace NDarray;

int dideal_recon_2D_CG( 
	Array< float,2 >&kx, 
	Array< float,2 >&ky, 
	Array< complex<float>,2 >&kdata,
	Array< float,3 >&fieldmap,
	Array< float,2 >&ktimes,
	float *freqs, 
	int Ns, 
	float lambda_space,
	float lambda_time,
	int max_iter);

int dideal_recon_2D_Gradient( 
	Array< float,2 >&kx, 
	Array< float,2 >&ky, 
	Array< complex<float>,2 >&kdata,
	Array< float,3 >&fieldmap,
	Array< float,2 >&ktimes,
	float *freqs, 
	int Ns, 
	float lambda_space,
	float lambda_time,
	float lambda_lowrank,
	int max_iter);

void lowrank_thresh( Array< complex<float>, 4> &image, float thresh);

int main( int argc, char **argv){

	// Hard Coded
	int Nt = 20;
	int Nr = 6692*2;
	int Ny = 128;
	int Nx = 128;
	int Ns = 3;
	int max_iter = 50;
	float lambda_space= 0.000;
	float lambda_time = 0.000;
	float lambda_lowrank = 0.000;
	float *freqs = new float[16];
	freqs[0]=-42;
	freqs[1]=-653;
	freqs[2]=-462;
	freqs[3]=-302;
	
#define float_flag(name,val)    }else if(strcmp(name,argv[pos]) == 0){ pos++; val = atof(argv[pos]);
#define int_flag(name,val)    }else if(strcmp(name,argv[pos]) == 0){ pos++; val = atoi(argv[pos]);
	for(int pos=0; pos < argc; pos++){
		if (strcmp("-h", argv[pos] ) == 0) {

			int_flag("-Nt",Nt);
			int_flag("-Nr",Nr);
			int_flag("-Ny",Ny);
			int_flag("-Nx",Nx);
			int_flag("-Ns",Ns);
			int_flag("-max_iter",max_iter);
			float_flag("-lambda_space",lambda_space);
			float_flag("-lambda_time",lambda_time);
			float_flag("-lambda_lowrank",lambda_lowrank);
			float_flag("-f0",freqs[0]);
			float_flag("-f1",freqs[1]);
			float_flag("-f2",freqs[2]);
			float_flag("-f3",freqs[3]);
			float_flag("-f4",freqs[4]);
			float_flag("-f5",freqs[5]);
			float_flag("-f6",freqs[6]);
		}
	}
	
	cout << "Imaging Parameters" << endl;
	cout << "  Nx = " << Nx << endl;
	cout << "  Ny = " << Ny << endl;
	cout << "  Nt = " << Nt << endl;
	cout << "  Ns = " << Ns << endl;
	cout << "  Nr = " << Nr << endl;
	cout << "Recon Parameters" << endl;
	for(int sp = 0; sp <Ns; sp++){
	cout << "  f" << sp << " = " << freqs[sp] << endl;
	}
	cout << "  lambda_space = " << lambda_space << endl;
	cout << "  lambda_time = " << lambda_time << endl;
	cout << "  lambda_lowrank = " << lambda_lowrank << endl;
	cout << "  max_iter = " << max_iter << endl;
	
	// Allocate Memory and Copy Values
	cout << "Kx" << endl << flush;
	Array< float,2>kx(Nr,Nt,ColumnMajorArray<2>());
	ArrayRead(kx,"Kx.dat");
	kx*=(float)1.0 / (float)Nx;

	cout << "Ky" << endl << flush;
	Array< float,2>ky(Nr,Nt,ColumnMajorArray<2>());
	ArrayRead(ky,"Ky.dat");
	ky*=(float)1.0 / (float)Ny;
	
	cout << "Kt" << endl << flush;
	Array< float,2>kt(Nr,Nt,ColumnMajorArray<2>());
	ArrayRead(kt,"Kt.dat");
	
	cout << "Kdata" << endl << flush;
	Array< complex<float> ,2>kdata(Nr,Nt,ColumnMajorArray<2>());
	ArrayRead(kdata,"Kdata.dat");
	
	
	cout << "Fieldmap" << endl << flush;
	Array< float,3>fieldmap(Nx,Ny,Nt,ColumnMajorArray<3>());
	ArrayRead(fieldmap,"FieldMap.dat");

//  No CG for comparison
	if(lambda_lowrank == 0){
		dideal_recon_2D_CG( kx, ky, kdata, fieldmap, kt, freqs, Ns,lambda_space,lambda_time,max_iter);
	}else{
		dideal_recon_2D_Gradient( kx, ky, kdata, fieldmap, kt, freqs, Ns,lambda_space,lambda_time,lambda_lowrank,max_iter);
	}

	return(0);
}

// Generate Images from Kdata
void transpose_dideal(
	const Array< complex<float>,2 > &diff_data,
	Array< complex<float>,4 > &X,
	const Array< float, 2 > &kx,
	const Array< float, 2 > &ky,
	const Array< float, 2 > &kt,
	const Array< float, 3 > &fieldmap,
	float *freqs){

	int Ns = X.length(fourthDim);
	int Nt = diff_data.length(secondDim); // Time Frame
	int Nr = diff_data.length(firstDim);  // Readout position

	// Spatial Coordinates
	int Nx = fieldmap.length(firstDim);
	int Ny = fieldmap.length(secondDim);
	float cx = (float)Ny/2.0;
	float cy = (float)Nx/2.0;
	const float pic=-6.28318530718;

	X = 0;

	float fov = (float)(Nx*Nx)/(4.0);
	// Loop Kx/Ky	
	tictoc T;
	T.tic();

#pragma omp parallel for schedule(static,1)
	for(int j=0; j<Ny; j++){

		// cout << "," << j << flush;
		float y = (float)j - cy;

		for(int i=0; i<Nx; i++){
			float x = (float)i - cx;
			float rad = x*x + y*y;
			if( rad > fov) continue;

			for(int species=0; species< Ns; species++){
				for(int t=0; t< Nt; t++){
					float Pf = freqs[species]+fieldmap(i,j,t);
					complex<float>temp(0.0,0.0);
					for(int rpos=0; rpos< Nr; rpos++){
						// Loop over image
						complex<float>basis = polar<float>((float)1.0, pic*(kx(rpos,t)*x + ky(rpos,t)*y + kt(rpos,t)*Pf));
						temp+= ( diff_data(rpos,t)*basis);
					}
					X(i,j,t,species)=temp;	// This takes all the time
				}// Nt
			}
		}
	}//Image Loop
	cout << "Transpose Took = " << T << endl;
}



// Generate Images from Kdata
void forward_dideal(
	Array< complex<float>,2 > &diff_data,
	const Array< complex<float>,4 > &X,
	const Array< float, 2 > &kx,
	const Array< float, 2 > &ky,
	const Array< float, 2 > &kt,
	const Array< float, 3 > &fieldmap,
	float *freqs){

	int Ns = X.length(fourthDim);
	int Nt = diff_data.length(secondDim); // Time Frame
	int Nr = diff_data.length(firstDim);  // Readout position

	// Spatial Coordinates
	int Nx = fieldmap.length(firstDim);
	int Ny = fieldmap.length(secondDim);
	float cx = (float)Ny/2.0;
	float cy = (float)Nx/2.0;
	const float pic= 6.28318530718;

	diff_data = 0;

	float fov = (float)(Nx*Nx)/(4.0);
	// Loop Kx/Ky	
	
	tictoc T;
	T.tic();

#pragma omp parallel for 
	for(int rpos=0; rpos< Nr; rpos++){
		for(int t=0; t< Nt; t++){


			// Position in k-t space
			float Kx = kx(rpos,t);
			float Ky = ky(rpos,t);
			float Kt = kt(rpos,t);

			// Loop over image
			for(int j=0; j<Ny; j++){
				float y = ((float)j - cy);
				float Py = y*Ky;
				for(int i=0; i<Nx; i++){
					float x =((float)i - cx);
					float Px = x*Kx;

					float rad = x*x + y*y;
					if( rad > fov) continue;
					for(int species=0; species< Ns; species++){
						complex<float>basis = polar<float>((float)1.0, pic*(Px + Py + Kt*( freqs[species]+fieldmap(i,j,t) )));
						diff_data(rpos,t) += (X(i,j,t,species)*basis);
					}
			}}//Image Loop

	}}// K-t  + Species Looop
	cout << "Forward Took = " << T << endl;
}


int dideal_recon_2D_CG( 
	Array< float,2 >&kx, 
	Array< float,2 >&ky, 
	Array< complex<float>,2 >&kdata,
	Array< float,3 >&fieldmap,
	Array< float,2 >&ktimes,
	float *freqs, 
	int Ns, 
	float lambda_space,
	float lambda_time,
	int max_iter){

	// Inputs
	//   kx      	[Nr,Nt ]          = Matrix with Nt readouts of length Nr 
	//   ky      	[Nr,Nt ]          = Matrix with Nt readouts of length Nr 
	//   kdata   	[Nr,Nt ]   		  = Matrix with Nt readouts of length Nr x Coils
	//   fieldmap	[ResX,ResY,Nt]    = Fieldmap
	//   kt      	[Nr,Nt]           = Readout times
	//   freqs   	[Ns x 1]          = Offsets for Species
	//   max_iter   [scalar]     		maximum iterations

	
	// Dimensions of Problem
	int Nr = ky.length(firstDim);
	int Nt = ky.length(secondDim);
	int Ny = fieldmap.length(firstDim);
	int Nx = fieldmap.length(secondDim);
	cout << "Array Size = " << Nx << " x " << Ny << " x " << Nt << " x " << Ns << endl;

	// ------------------------------------
	// Iterative Soft Thresholding  x(n+1)=  thresh(   x(n) - E*(Ex(n) - d)  )
	//  Designed to not use memory
	// Uses gradient descent x(n+1) = x(n) - ( R'R ) / ( R'E'E R) * Grad  [ R = E'(Ex-d)]
	// ------------------------------------

	// Final Image Solution
	Array< complex<float>,4 >X(Nx,Ny,Nt,Ns,ColumnMajorArray<4>());
	X = 0;

	// Residue 	
	Array< complex<float>,4 >R(Nx,Ny,Nt,Ns,ColumnMajorArray<4>());
	R = 0;;

	// Residue 	
	Array< complex<float>,4 >Reg(Nx,Ny,Nt,Ns,ColumnMajorArray<4>());
	Reg = 0;

	// Temp variable for E'ER 
	Array< complex<float>,4 >P(Nx,Ny,Nt,Ns,ColumnMajorArray<4>());
	P = 0;;

	// Temp variable for E'ER 
	Array< complex<float>,2 >diff_data(Nr,Nt,ColumnMajorArray<2>());
	diff_data = 0;
	


	// RHS (CG)
	Array< complex<float>,4 >LHS(Nx,Ny,Nt,Ns,ColumnMajorArray<4>());
	
	cout << "Init CG" << endl;
	
	LHS= complex<float>(0,0);
	R= complex<float>(0,0);
	P= complex<float>(0,0);
	
	transpose_dideal(kdata,R,kx,ky,ktimes,fieldmap,freqs);
	R = -R;
	P = R;

	// Conjugate Gradient
	double error0=0.0;
	float reg_scale2=0.0; 
	for(int iteration =0; iteration< max_iter; iteration++){
		
		cout << "\nIteration = " << iteration << endl;
		
		diff_data = complex<float>(0,0);
		forward_dideal(diff_data,P,kx,ky,ktimes,fieldmap,freqs);
		
		LHS = complex<float>(0,0);
		transpose_dideal(diff_data,LHS,kx,ky,ktimes,fieldmap,freqs);

		// Convolve with TV
		for(int sp=0; sp<Ns; sp++){
			for(int t=0;t<Nt;t++){
				for(int j=0;j<Ny;j++){
					for(int i=0;i<Nx;i++){
						
						// Space
						Reg(i,j,t,sp)=complex<float>(4.0,0)*P(i,j,t,sp);
						Reg(i,j,t,sp)-=P( (i+1+Nx)%Nx,j,t,sp);
						Reg(i,j,t,sp)-=P( (i-1+Nx)%Nx,j,t,sp);
						Reg(i,j,t,sp)-=P( i,(j+1+Ny)%Ny,t,sp);
						Reg(i,j,t,sp)-=P( i,(j-1+Ny)%Ny,t,sp);
						
						//Time
						Reg(i,j,t,sp) +=lambda_time*complex<float>(2.0,0)*P(i,j,t,sp);
						Reg(i,j,t,sp) -=lambda_time*P(i,j,(Nt+t-1)%Nt,sp);
						Reg(i,j,t,sp) -=lambda_time*P(i,j,(Nt+t+1)%Nt,sp);
						
						
		}}}}			



		if(iteration==0){
			error0 = ArrayEnergy(R);
			reg_scale2 = lambda_space*sqrt( error0 );
		}
		Reg *= reg_scale2;
		LHS += Reg; 

		complex< float> sum_R0_R0(0.0,0.0);
		complex< float> sum_R_R(0.0,0.0);
		complex< float> sum_P_LHS(0.0,0.0);
		
		// Calc R'R and P'*LHS
		for(int sp=0; sp<Ns; sp++){
			for(int t=0;t<Nt;t++){
				for(int j=0;j<Ny;j++){
					for(int i=0;i<Nx;i++){			
						sum_R0_R0 += norm( R(i,j,t,sp));
						sum_P_LHS += conj( P(i,j,t,sp))*LHS(i,j,t,sp);
		}}}}
		complex< float> scale = sum_R0_R0 / sum_P_LHS; 


		// Calc R'R and P'*LHS
		for(int sp=0; sp<Ns; sp++){
			for(int t=0;t<Nt;t++){
				for(int j=0;j<Ny;j++){
					for(int i=0;i<Nx;i++){			
						X(i,j,t,sp) += ( scale*P(i,j,t,sp) );
						R(i,j,t,sp) -= ( scale*LHS(i,j,t,sp) );
						sum_R_R += norm( R(i,j,t,sp) );
		}}}}
		
		cout << "Sum R'R = " << sum_R_R << endl;
		complex< float> scale2 = sum_R_R / sum_R0_R0; 

		// Take step size
		for(int sp=0; sp<Ns; sp++){
			for(int t=0;t<Nt;t++){
				for(int j=0;j<Ny;j++){
					for(int i=0;i<Nx;i++){			
						P(i,j,t,sp) += R(i,j,t,sp) + scale2*P(i,j,t,sp);
		}}}}

		char fname[80];
		for(int sp=0; sp<Ns; sp++){
			sprintf(fname,"Species_%d.dat",sp);
			Array< complex<float>,3> TempS = X( Range::all(),Range::all(),Range::all(),sp);
			ArrayWrite( TempS,fname);
		}
	}


	return(0);
}

int dideal_recon_2D_Gradient( 
	Array< float,2 >&kx, 
	Array< float,2 >&ky, 
	Array< complex<float>,2 >&kdata,
	Array< float,3 >&fieldmap,
	Array< float,2 >&ktimes,
	float *freqs, 
	int Ns, 
	float lambda_space,
	float lambda_time,
	float lambda_lowrank,
	int max_iter){

	// Inputs
	//   kx      	[Nr,Nt ]          = Matrix with Nt readouts of length Nr 
	//   ky      	[Nr,Nt ]          = Matrix with Nt readouts of length Nr 
	//   kdata   	[Nr,Nt ]   		  = Matrix with Nt readouts of length Nr x Coils
	//   fieldmap	[ResX,ResY,Nt]    = Fieldmap
	//   kt      	[Nr,Nt]           = Readout times
	//   freqs   	[Ns x 1]          = Offsets for Species
	//   max_iter   [scalar]     		maximum iterations

	
	// Dimensions of Problem
	int Nr = ky.length(firstDim);
	int Nt = ky.length(secondDim);
	int Ny = fieldmap.length(firstDim);
	int Nx = fieldmap.length(secondDim);
	cout << "Array Size = " << Nx << " x " << Ny << " x " << Nt << " x " << Ns << endl;

	// ------------------------------------
	// Iterative Soft Thresholding  x(n+1)=  thresh(   x(n) - E*(Ex(n) - d)  )
	//  Designed to not use memory
	// Uses gradient descent x(n+1) = x(n) - ( R'R ) / ( R'E'E R) * Grad  [ R = E'(Ex-d)]
	// ------------------------------------

	// Final Image Solution
	Array< complex<float>,4 >X(Nx,Ny,Nt,Ns,ColumnMajorArray<4>());
	Array< complex<float>,4 >R(Nx,Ny,Nt,Ns,ColumnMajorArray<4>());
	Array< complex<float>,4 >Reg(Nx,Ny,Nt,Ns,ColumnMajorArray<4>());
	Array< complex<float>,2 >diff_data(Nr,Nt,ColumnMajorArray<2>());
	Array< complex<float>,4 >P(Nx,Ny,Nt,Ns,ColumnMajorArray<4>());
	

	R= complex<float>(0,0);
	X= complex<float>(0,0);
	
	
	// Conjugate Gradient
	double error0=0.0;
	float reg_scale2=0.0; 
	for(int iteration =0; iteration< max_iter; iteration++){
		
		cout << "Iteration = " << iteration << endl;
		
		// Get data
		diff_data = complex<float>(0,0);
		forward_dideal(diff_data,X,kx,ky,ktimes,fieldmap,freqs);
		
		// Difference
		diff_data -= kdata;
				
		// Get Residue
		R= complex<float>(0,0);
		transpose_dideal(diff_data,R,kx,ky,ktimes,fieldmap,freqs);

		// Convolve with TV
		for(int sp=0; sp<Ns; sp++){
			for(int t=0;t<Nt;t++){
				for(int j=0;j<Ny;j++){
					for(int i=0;i<Nx;i++){
						
						// Space
						Reg(i,j,t,sp)=lambda_space*complex<float>(4.0,0)*X(i,j,t,sp);
						Reg(i,j,t,sp)-=lambda_space*X( (i+1+Nx)%Nx,j,t,sp);
						Reg(i,j,t,sp)-=lambda_space*X( (i-1+Nx)%Nx,j,t,sp);
						Reg(i,j,t,sp)-=lambda_space*X( i,(j+1+Ny)%Ny,t,sp);
						Reg(i,j,t,sp)-=lambda_space*X( i,(j-1+Ny)%Ny,t,sp);
						
						//Time
						Reg(i,j,t,sp) +=lambda_time*complex<float>(2.0,0)*X(i,j,t,sp);
						Reg(i,j,t,sp) -=lambda_time*X(i,j,(Nt+t-1)%Nt,sp);
						Reg(i,j,t,sp) -=lambda_time*X(i,j,(Nt+t+1)%Nt,sp);
						
						
		}}}}			

		if(iteration==0){
			error0 = ArrayEnergy(R);
			reg_scale2 = sqrt( error0 );
		}
		Reg *= reg_scale2;
		R += Reg; 
		
		
		// Now get Scale
		diff_data = complex<float>(0,0);
		forward_dideal(diff_data,R,kx,ky,ktimes,fieldmap,freqs);
		P = complex<float>(0,0);
		transpose_dideal(diff_data,P,kx,ky,ktimes,fieldmap,freqs);
		
		cout << "Energy = " << ArrayEnergy(R)/error0 << endl;
		
		P *= conj(R);
		cout << "RhR = " << ArrayEnergy(R) << endl;
		cout << "RhP = " << sum(P) << endl;
		
		
		complex<float>scale = ArrayEnergy(R)/sum(P);
		cout << "Scale = " << scale << endl;
		
		R *= scale;
		X -= R;
		
		
		{
			char fname[80];
			for(int sp=0; sp<Ns; sp++){
				sprintf(fname,"Species_%d.dat.slice",sp);
				Array< complex<float>,2> TempS = X( Range::all(),Range::all(),(int)(Nt/2),sp);	
				ArrayWriteMagAppend(TempS,fname);
			}
		}
		
		if(lambda_lowrank > 0){
			lowrank_thresh(X,lambda_lowrank);
		}
			
		{
			char fname[80];
			for(int sp=0; sp<Ns; sp++){
				sprintf(fname,"Species_%d.dat.slice",sp);
				Array< complex<float>,2> TempS = X( Range::all(),Range::all(),(int)(Nt/2),sp);	
				ArrayWriteMagAppend(TempS,fname);
			}
		}
	}

	char fname[80];
	for(int sp=0; sp<Ns; sp++){
		sprintf(fname,"Species_%d.dat",sp);
		Array< complex<float>,3> TempS = X( Range::all(),Range::all(),Range::all(),sp);
		ArrayWrite( TempS,fname);
	}

	return(0);
}

void lowrank_thresh( Array< complex<float>,4> &image, float thresh){

	int Nx = image.length(firstDim);
	int Ny = image.length(secondDim);
	int Nt = image.length(thirdDim);
	int Ns = image.length(fourthDim);
	int Np = Nx*Ny*Ns;
	
	// Copy image into matrix
	arma::cx_mat A;
	A.zeros(Nx*Ny*Ns,Nt);
	
	for(int t =0; t < Nt; t++){
		int count = 0;

		for(int s =0; s<Ns; s++){
		for(int j =0; j<Ny; j++){
		for(int i =0; i<Nx; i++){
			A(count,t) = image(i,j,t,s);	
			count++;
		}}}
	}
	
	// SVD
	cout << "Svd" << endl;
	arma::cx_mat U;
	arma::vec s;
	arma::cx_mat V;
	arma::svd(U,s,V,A);
	
	arma::mat S;
	S.zeros(Np,Nt); // Pixels x Coils
	
	cout << "Thresh with " << thresh << endl;
	double smax = thresh*max(s);
	for(int pos =0; pos< min(Nt,Np); pos++){
		S(pos,pos)=   max( s(pos) - smax, 0.0 );
	}
	
	A = U*S*V.t(); 
	
	for(int t =0; t < Nt; t++){
		int count = 0;

		for(int s =0; s<Ns; s++){
		for(int j =0; j<Ny; j++){
		for(int i =0; i<Nx; i++){
			image(i,j,t,s) = A(count,t);	
			count++;
		}}}
	}
}

