#include "clear.h"
#include "io_templates.hpp"

using arma::cx_fmat;
using arma::fmat;
using arma::fvec;
using arma::uvec;
using namespace std;
using namespace NDarray;

// ----------------------
// Help Message
// ----------------------
void LOWRANKCOIL::help_message(void){
	cout << "----------------------------------------------" << endl;
	cout << "  Control for Low Rank Operators " << endl;
	cout << "----------------------------------------------" << endl;
	
	help_flag("-clear_nx []","kernel size in x");
	help_flag("-clear_ny []","kernel size in y");
	help_flag("-clear_ns []","kernel size in z");
	help_flag("-clear_alpha[]","regularization factor");
	
}

LOWRANKCOIL::LOWRANKCOIL(){

  // Kernel Size Defaults 
  block_size_x = 8;
  block_size_y = 8;
  block_size_z = 8;

  clear_alpha = 0.01;
  debug =0;
  smax = 1.0;
}
  

LOWRANKCOIL::LOWRANKCOIL(int numarg, char **pstring){
  
  // Kernel Size Defaults 
  block_size_x = 4;
  block_size_y = 4;
  block_size_z = 4;

  clear_alpha = 0.01;
  debug =0;
  smax = 1.0;

#define trig_flag(num,name,val)   }else if(strcmp(name,pstring[pos]) == 0){ val = num; 
#define float_flag(name,val)  }else if(strcmp(name,pstring[pos]) == 0){ pos++; val = atof(pstring[pos]); 
#define int_flag(name,val)    }else if(strcmp(name,pstring[pos]) == 0){ pos++; val = atoi(pstring[pos]);

  for(int pos=0; pos < numarg; pos++){
  
  	if(strcmp("-h", pstring[pos] ) == 0) {
	  	
		int_flag("-clear_nx",block_size_x);
		int_flag("-clear_ny",block_size_y);
		int_flag("-clear_nz",block_size_z);
		float_flag("-clear_alpha",clear_alpha);
	}
  }
}    

//-----------------------------------------------------
//Clear Threshold Estimate
//
//  Just estimate the max singular value over the matrix  
//-----------------------------------------------------
void LOWRANKCOIL::update_threshold( Array< Array<complex<float>,3>,3 > &image){
	
	typedef Array<complex<float>,3> Complex3D;
	
	
	float *stemp = new float[image(0,0,0).extent(thirdDim)];
	
	
	for(int t=0; t< image.length(firstDim); t++){
	for(int e=0; e< image.length(secondDim); e++){
		
	#pragma omp parallel for
	for( int k=0; k < image(0,0,0).extent(thirdDim); k+=block_size_z){
		stemp[k] = 0.0;
		
		cx_fmat A;
		A.zeros(block_size_x*block_size_y*block_size_z,image.extent(thirdDim)); // Pixels x Coils
	
		fmat S;
		S.zeros(block_size_x*block_size_y*block_size_z,image.extent(thirdDim)); // Pixels x Coils
		
		for( int j=0; j < image(0,0,0).extent(secondDim); j+=block_size_y){
		for( int i=0; i < image(0,0,0).extent(firstDim); i+=block_size_x){
		
			int count = 0;
			for(int kk=k; kk < k+block_size_z; kk++){
			for(int jj=j; jj < j+block_size_y; jj++){
			for(int ii=i; ii < i+block_size_x; ii++){
				for(int coil=0; coil < image.extent(thirdDim); coil++){
					A(count,coil) = image(t,e,coil)(ii,jj,kk);
				}
				count++;
			}}}
			
			// SVD
			fvec s;
  			arma::svd(s,A);
			
			stemp[k]= ( s(0) > stemp[k] ) ? ( s(0) ) : ( stemp[k] );
	}}}
	
	}}
	
	smax = stemp[0];
	for( int k=0; k < image(0,0,0).extent(thirdDim); k+=block_size_z){
		smax = ( smax > stemp[k] ) ? ( smax ) : ( stemp[k] );
	}
	
	delete [] stemp;

	cout << "Max singular value is " << smax << endl;

}


//-----------------------------------------------------
//Clear Threshold Estimate
//
//  Just estimate the max singular value over the matrix  
//-----------------------------------------------------
void LOWRANKCOIL::update_threshold( Array< Array<complex<float>,3>,2 > &image, int dim){
	
	
	typedef Array<complex<float>,3> Complex3D;
	
	// Shorthand
	int Nt =image.extent(firstDim);
	int Ne =image.extent(secondDim);
	int Nx =image(0,0).extent(firstDim);
	int Ny =image(0,0).extent(secondDim);
	int Nz =image(0,0).extent(thirdDim);
	
	// Blocks shouldn't be larger than the dimension
	block_size_x = ( block_size_x > Nx) ? ( Nx ) : ( block_size_x );
	block_size_y = ( block_size_y > Ny) ? ( Ny ) : ( block_size_y );
	block_size_z = ( block_size_z > Nz) ? ( Nz ) : ( block_size_z );
	
	
	int N = image.extent(dim);
	int Np = image.extent( (dim+1)%2)*block_size_x*block_size_y*block_size_z; 
	
	cout << "Actual Block Size = " << block_size_x << " x " << block_size_y << " x " << block_size_z << endl;	
	cout << "Getting Low Rank threshold (N=" << N << ")(Np = " << Np << ")" << endl;
	smax = 0.0;
	
//	int block_x_start=0;
//	int block_y_start=0;
//	int block_z_start=0;
	
	int block_x_stop= (int)( Nx / block_size_x );
	int block_y_stop= (int)( Ny / block_size_y );
	int block_z_stop= (int)( Nz / block_size_z );
	
	int total_blocks = block_x_stop * block_y_stop * block_z_stop;
	cout << "Total Block Size" << total_blocks << " ( " << block_x_stop << "," << block_y_stop << "," << block_z_stop << ")" << endl;
	
	#pragma omp parallel for
	for( int block = 0; block < total_blocks; block++){
		
		// Nested parallelism workaround
		int i = (int)( block % block_size_x);
		int temp = (int)( block / block_size_x);
		int j = (int)( temp % block_size_y );
		int k = (int)( temp / block_size_y );
		
		cout << "Block = " << block << "k=" << k << ",j=" << j << ",i=" << endl << flush;
			
		cx_fmat A;
		A.zeros(Np,N); // Pixels x N
	
		fmat S;
		S.zeros(Np,N); // Pixels x N
		
		
		
			//-----------------------------------------------------
			//   Block section
			//-----------------------------------------------------
			switch(dim){
				case(0):{
					for(int t=0; t < Nt; t++){
					int count=0;
					for(int e=0; e < Ne; e++){
					for(int kk=k; kk < k+block_size_z; kk++){
					for(int jj=j; jj < j+block_size_y; jj++){
					for(int ii=i; ii < i+block_size_x; ii++){
						A(count,t) = image(t,e)(ii,jj,kk);
						count++;
					}}}}}
				}break;
				
				case(1):{
					for(int e=0; e < Ne; e++){
					int count=0;
					for(int t=0; t < Nt; t++){
					for(int kk=k; kk < k+block_size_z; kk++){
					for(int jj=j; jj < j+block_size_y; jj++){
					for(int ii=i; ii < i+block_size_x; ii++){
						A(count,e) = image(t,e)(ii,jj,kk);
						count++;
					}}}}}
				}break;
			}// Switch Dim	
				
			// SVD
			fvec s;
  			arma::svd(s,A);
			
			#pragma omp critical
			{
				smax = max(  s(0),smax);
			}
			
	}// Block (threaded)
	cout << "Max singular value is " << smax << endl;
}




//-----------------------------------------------------
//  Clear Threshold Call 
//-----------------------------------------------------
void LOWRANKCOIL::thresh( Array< Array<complex<float>,3>,2 > &image, int dim){
		
	if(clear_alpha==0.0){
		return;
	}	
	
	// Shorthand
	int Nt =image.extent(firstDim);
	int Ne =image.extent(secondDim);
	int Nx =image(0,0).extent(firstDim);
	int Ny =image(0,0).extent(secondDim);
	int Nz =image(0,0).extent(thirdDim);
	
//	int step_size_x = (int)ceil((float)block_size_x);
//	int step_size_y = (int)ceil((float)block_size_y);
//	int step_size_z = (int)ceil((float)block_size_z);
	
	float block_iter = 4;
	
	int N = image.extent(dim);
	int Np = image.extent( (dim+1)%2)*block_size_x*block_size_y*block_size_z; 
	
	cout << " Low Rank threshold (N=" << N << ")(Np = " << Np << ")" << endl;
	
		
	for(int iteration= 0; iteration < block_iter; iteration++){	
		
	int shiftx = rand() % block_size_x - block_size_x/2;
	int shifty = rand() % block_size_y - block_size_y/2;
	int shiftz = rand() % block_size_z - block_size_z/2;
	
	int block_x_stop= (int)( Nx / block_size_x );
	int block_y_stop= (int)( Ny / block_size_y );
	int block_z_stop= (int)( Nz / block_size_z );
	
	int total_blocks = block_x_stop * block_y_stop * block_z_stop;
	cout << "Total Block Size" << total_blocks << " ( " << block_x_stop << "," << block_y_stop << "," << block_z_stop << ")" << endl;
		
	#pragma omp parallel for
	for( int block = 0; block < total_blocks; block++){
		
		// Nested parallelism workaround
		int i = (int)( block % block_size_x);
		int temp = (int)( block / block_size_x);
		int j = (int)( temp % block_size_y );
		int k = (int)( temp / block_size_y );
		
		cout << "Block = " << block << "k=" << k << ",j=" << j << ",i=" << i << endl << flush;
		
		cx_fmat A;
		A.zeros(Np,N); // Pixels x Coils
	
		cx_fmat U;
		U.zeros(Np,Np); // Pixels x Pixels
	    
		fmat S;
		S.zeros(Np,N); // Pixels x Coils
	    		
		cx_fmat V;
		V.zeros(N,N); // Coils x Coils
	    
	
			//-----------------------------------------------------
			//   Block Gather
			//-----------------------------------------------------
			switch(dim){
				case(0):{
					for(int t=0; t < Nt; t++){
					int count=0;
					for(int e=0; e < Ne; e++){
					for(int kk=(k+shiftz); kk < (k+block_size_z+shiftz); kk++){
					for(int jj=(j+shifty); jj < (j+block_size_y+shifty); jj++){
					for(int ii=(i+shiftx); ii < (i+block_size_x+shiftx); ii++){
						int px = (ii + Nx)% Nx;
						int py = (jj + Ny)% Ny;
						int pz = (kk + Nz)% Nz;
						A(count,t) = image(t,e)(px,py,pz);
						count++;
					}}}}}
				}break;
				
				case(1):{
					for(int e=0; e < Ne; e++){
					int count=0;
					for(int t=0; t < Nt; t++){
					for(int kk=(k+shiftz); kk < (k+block_size_z+shiftz); kk++){
					for(int jj=(j+shifty); jj < (j+block_size_y+shifty); jj++){
					for(int ii=(i+shiftx); ii < (i+block_size_x+shiftx); ii++){
						int px = (ii + Nx)% Nx;
						int py = (jj + Ny)% Ny;
						int pz = (kk + Nz)% Nz;
						A(count,e) = image(t,e)(px,py,pz);
						count++;
					}}}}}
				}break;
			}// Switch Dim	
			
			
			// --------------------------------------------------------
			// SVD Threshold
			// -------------------------------------------------------- 
			fvec s;
  			arma::svd(U,s,V,A);
								
			S.zeros();
			for(int pos =0; pos< min(N,Np); pos++){
				S(pos,pos)=   max( s(pos) - smax*clear_alpha/block_iter, 0.0f );
			}
												
			// Reconstruct 
			cx_fmat X = U*S*trans(V);
			
						
			// --------------------------------------------------------
			//  Block Scatter
			// -------------------------------------------------------- 
			
			switch(dim){
				case(0):{
					for(int t=0; t < Nt; t++){
					int count=0;
					for(int e=0; e < Ne; e++){
					for(int kk=(k+shiftz); kk < (k+block_size_z+shiftz); kk++){
					for(int jj=(j+shifty); jj < (j+block_size_y+shifty); jj++){
					for(int ii=(i+shiftx); ii < (i+block_size_x+shiftx); ii++){
						int px = (ii + Nx)% Nx;
						int py = (jj + Ny)% Ny;
						int pz = (kk + Nz)% Nz;
						image(t,e)(px,py,pz) = X(count,t);
						count++;
					}}}}}
				}break;
				
				case(1):{
					for(int e=0; e < Ne; e++){
					int count=0;
					for(int t=0; t < Nt; t++){
					for(int kk=(k+shiftz); kk < (k+block_size_z+shiftz); kk++){
					for(int jj=(j+shifty); jj < (j+block_size_y+shifty); jj++){
					for(int ii=(i+shiftx); ii < (i+block_size_x+shiftx); ii++){
						int px = (ii + Nx)% Nx;
						int py = (jj + Ny)% Ny;
						int pz = (kk + Nz)% Nz;
						image(t,e)(px,py,pz) = X(count,e);
						count++;
					}}}}}
				}break;
			}// Switch Dim	
			
			
	}// Block Loop
	
	}//iteration
}


//-----------------------------------------------------
//  Clear Threshold Call 
//-----------------------------------------------------
void LOWRANKCOIL::thresh( Array< Array<complex<float>,3>,3 > &image, Array< Array<complex<float>,3>,3 > &temp){
		
	if(clear_alpha==0.0){
		return;
	}	
	
	// Copy to temp location
	swap(image, temp);
	for( Array< Array<complex<float>,3>,3 >::iterator miter=image.begin(); miter != image.end(); miter++){
		*miter = complex<float>(0.0,0.0);
	}			
	
	int Ncoils = image.extent(thirdDim);
	
	int shiftx;
	int shifty;
	int shiftz;
	
	int rand_shiftx = rand() % block_size_x - block_size_x/2;
	int rand_shifty = rand() % block_size_y - block_size_y/2;
	int rand_shiftz = rand() % block_size_z - block_size_z/2;
		
	int Nx = image(0,0,0).extent(firstDim);
	int Ny = image(0,0,0).extent(secondDim);
	int Nz = image(0,0,0).extent(thirdDim);
	
	int step_size_x = (int)ceil((float)block_size_x);
	int step_size_y = (int)ceil((float)block_size_y);
	int step_size_z = (int)ceil((float)block_size_z);
		
	//cout << "Block size = " << block_size_x << " x " << block_size_y << " x " << block_size_z << endl;
	//cout << "Shifts = " << shiftx << " x " << shifty << " x " << shiftz << endl;
	//cout << "Steps = " << step_size_x << " x " << step_size_y << " x " << step_size_z << endl;
	
	for(int offset=0; offset <2; offset++){
		if(offset ==0){
			shiftx = -(int)floor(block_size_x/2)+rand_shiftx;
			shifty = -(int)floor(block_size_y/2)+rand_shifty;
			shiftz = -(int)floor(block_size_z/2)+rand_shiftz;
		}else{
			shiftx = rand_shiftx;
			shifty = rand_shifty;
			shiftz = rand_shiftz;
		}
		
		
	for(int t=0; t < image.length(firstDim); t++){
	for(int e=0; e < image.length(secondDim); e++){
			
			
	#pragma omp parallel for
	for( int k=0; k < Nz; k+=step_size_z){
		
		cx_fmat A;
		A.zeros(block_size_x*block_size_y*block_size_z,Ncoils); // Pixels x Coils
	
		cx_fmat U;
		U.zeros(block_size_x*block_size_y*block_size_z,block_size_x*block_size_y*block_size_z); // Pixels x Pixels
	    
		fmat S;
		S.zeros(block_size_x*block_size_y*block_size_z,Ncoils); // Pixels x Coils
	    		
		cx_fmat V;
		V.zeros(Ncoils,Ncoils); // Coils x Coils
	    
		 		
		for( int j=0; j < Ny; j+=step_size_y){ 
		for( int i=0; i < Nx; i+=step_size_x){
		
			int count = 0;
			for(int kk=(k+shiftz); kk < (k+block_size_z+shiftz); kk++){
			for(int jj=(j+shifty); jj < (j+block_size_y+shifty); jj++){
			for(int ii=(i+shiftx); ii < (i+block_size_x+shiftx); ii++){
				
				int px = (ii + Nx)% Nx;
				int py = (jj + Ny)% Ny;
				int pz = (kk + Nz)% Nz;
				for(int coil=0; coil < Ncoils; coil++){
					A(count,coil) = temp(t,e,coil)(px,py,pz);
				}
				count++;
			}}}
			
			// SVD
			fvec s;
  			arma::svd(U,s,V,A);
								
			S.zeros();
			for(int pos =0; pos< Ncoils; pos++){
				S(pos,pos)=   max( s(pos) - smax*clear_alpha, 0.0f );
			}
												
			// Reconstruct 
			cx_fmat X = U*S*trans(V);
			
			// Copy Back in weighted Fashion
			count = 0;
			for(int kk=(k+shiftz); kk < (k+block_size_z+shiftz); kk++){
			for(int jj=(j+shifty); jj < (j+block_size_y+shifty); jj++){
			for(int ii=(i+shiftx); ii < (i+block_size_x+shiftx); ii++){
				int px = (ii + Nx)% Nx;
				int py = (jj + Ny)% Ny;
				int pz = (kk + Nz)% Nz;
				
				complex<float>w(0.5,0.0);
				for(int coil=0; coil < Ncoils; coil++){
					image(t,e,coil)(px,py,pz) += X(count,coil)*w;
				}
				count++;
			}}}

	}}}// spatial
	
	}}// t + e
	
	}//Block shifts
		  
		 		  
}

//-----------------------------------------------------
//  Clear Combine Call 
//-----------------------------------------------------
void LOWRANKCOIL::combine( Array< Array<complex<float>,3>,3 > &image,Array< Array<complex<float>,3>,2 > &temp){
	
	typedef Array< complex<float>,3> Complex3D;
	
	for( Array< Complex3D, 2>::iterator miter=temp.begin(); miter!=temp.end(); miter++){
		*miter = complex<float>(0.0,0.0);
	}
		
	int shiftx;
	int shifty;
	int shiftz;
	
	int rand_shiftx = 0;
	int rand_shifty = 0;
	int rand_shiftz = 0;
		
	int Nx = image(0,0,0).extent(firstDim);
	int Ny = image(0,0,0).extent(secondDim);
	int Nz = image(0,0,0).extent(thirdDim);
	int Ncoils=  image.length(thirdDim);
	
	int step_size_x = (int)ceil((float)block_size_x);
	int step_size_y = (int)ceil((float)block_size_y);
	int step_size_z = (int)ceil((float)block_size_z);
		
	//cout << "Block size = " << block_size_x << " x " << block_size_y << " x " << block_size_z << endl;
	//cout << "Shifts = " << shiftx << " x " << shifty << " x " << shiftz << endl;
	//cout << "Steps = " << step_size_x << " x " << step_size_y << " x " << step_size_z << endl;
	
	for(int offset=0; offset <1; offset++){
		if(offset ==0){
			shiftx = -(int)floor(block_size_x/2)+rand_shiftx;
			shifty = -(int)floor(block_size_y/2)+rand_shifty;
			shiftz = -(int)floor(block_size_z/2)+rand_shiftz;
		}else{
			shiftx = rand_shiftx;
			shifty = rand_shifty;
			shiftz = rand_shiftz;
		}
		
	for(int t=0; t< image.length(firstDim); t++){
	for(int e=0; e< image.length(secondDim); e++){
	
			
	#pragma omp parallel for
	for( int k=0; k < Nz; k+=step_size_z){
		
		cx_fmat A;
		A.zeros(block_size_x*block_size_y*block_size_z,Ncoils); // Pixels x Coils
	
		cx_fmat U;
		U.zeros(block_size_x*block_size_y*block_size_z,block_size_x*block_size_y*block_size_z); // Pixels x Pixels
	    
		fmat S;
		S.zeros(block_size_x*block_size_y*block_size_z,Ncoils); // Pixels x Coils
	    		
		cx_fmat V;
		V.zeros(Ncoils,Ncoils); // Coils x Coils
	    
		 		
		for( int j=0; j < Ny; j+=step_size_y){ 
		for( int i=0; i < Nx; i+=step_size_x){
		
			int count = 0;
			for(int kk=(k+shiftz); kk < (k+block_size_z+shiftz); kk++){
			for(int jj=(j+shifty); jj < (j+block_size_y+shifty); jj++){
			for(int ii=(i+shiftx); ii < (i+block_size_x+shiftx); ii++){
				
				int px = (ii + Nx)% Nx;
				int py = (jj + Ny)% Ny;
				int pz = (kk + Nz)% Nz;
				for(int coil=0; coil < image.extent(fourthDim); coil++){
					A(count,coil) = image(t,e,coil)(px,py,pz);
				}
				count++;
			}}}
			
			// SVD
			fvec s;
  			arma::svd(U,s,V,A);
								
			// Grab Largest Value
			cx_fmat A2 = A*V.cols(0,0);
						
			// Copy Back in weighted Fashion
			count = 0;
			for(int kk=(k+shiftz); kk < (k+block_size_z+shiftz); kk++){
			for(int jj=(j+shifty); jj < (j+block_size_y+shifty); jj++){
			for(int ii=(i+shiftx); ii < (i+block_size_x+shiftx); ii++){
				int px = (ii + Nx)% Nx;
				int py = (jj + Ny)% Ny;
				int pz = (kk + Nz)% Nz;
				
				complex<float>w(1.0,0.0);
				temp(t,e)(px,py,pz) += A2(count,0)*w;
				count++;
			}}}

	}}}//Spatial
	
	}}// t+e
	
	}
		  
		 		  
}


