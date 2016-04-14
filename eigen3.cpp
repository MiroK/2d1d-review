#include <iostream>
#include <cmath>
#include <ctime>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;

MatrixXd stiffness(unsigned int n){
  MatrixXd A = MatrixXd::Zero(n+1, n+1);
  double h = 1./n;

  A(0, 0) = 1;  
  for(unsigned int i = 1; i < n; i++)
    A(i, i) = 2./h;
  A(n, n) = 1;

  for(unsigned int i = 1; i < n-1; i++){
    A(i, i+1) = -1./h;
    A(i+1, i) = -1./h;
  }
  A(n-1, n) = -1./h;

  return A;
}


MatrixXd mass(unsigned int n){
  MatrixXd M = MatrixXd::Zero(n+1, n+1);
  double h = 1./n;

  M(0, 0) = 1;  
  for(unsigned int i = 1; i < n; i++)
    M(i, i) = 4*h/6;
  M(n, n) = 1;

  for(unsigned int i = 1; i < n-1; i++){
    M(i, i+1) = h/6;
    M(i+1, i) = h/6;
  }
  M(n-1, n) = h/6;

  return M;
}

// Eigen is not very fast...
int main()
{
  for(unsigned int i = 1; i < 11; i++){
    unsigned n = (unsigned int)pow(2, i);

    MatrixXd A = stiffness(n);
    MatrixXd M = mass(n);


    clock_t begin = clock();
    GeneralizedSelfAdjointEigenSolver<MatrixXd> es(A, M);
    clock_t end = clock();
    double elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    
    VectorXd eigenvalues= es.eigenvalues();
    double lmin = eigenvalues.minCoeff();
    double lmax = eigenvalues.maxCoeff();
    cout << "n = " << n << " lmin = " << lmin << " lmax " << lmax << " took " << elapsed_secs << endl;
  }
}

// g++ -I /path/to/eigen_include/eigen3 my_program.cpp -o my_program 

