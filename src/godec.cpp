#include <iostream>
#include "stbi/stb_image.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi/stb_image_write.c"
#include <eigen3/Eigen/Dense>
#include "cmdline.h"
//#include "ye/data.hpp"
typedef double val_t;
using namespace std;

const char* FILE_NAME     = "/home/ehara/mypic.png";
//const char* FILE_NAME     = "/home/ehara/akb48_2.png";
const char* X_FILE_NAME = "X.png";
const char* L_FILE_NAME = "L.png";
const char* S_FILE_NAME = "S.png";
const char* LpS_FILE_NAME = "LpS.png";
const char* SVD_FILE_NAME = "svd.png";

typedef Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic> emat_t;

void grayscale( int width, int height,int bpp, unsigned char* pixels,  double range = 1.0, unsigned char shift=0){
    for(int i=0;i<height;++i){
      for(int j=0;j<width;++j){
        int id = bpp*(j+width*i);
        unsigned int isum = 0;
        for(int k=0;k<bpp;++k) isum += pixels[id+k];
        unsigned char gray = isum/bpp*range;
        for(int k=0;k<bpp;++k) pixels[id+k] = gray + shift;
      }
    }
}    

template<class T>
void pixel2mat(int width, int height, int bpp,  const unsigned char * pixels, T& mat, double range=1.0, unsigned char shift = 0){
    mat.resize(height, width);
    for(int i=0;i<height;++i){
      for(int j=0;j<width;++j){
        int id = bpp*(j+width*i);
        unsigned int isum = 0;
        for(int k=0;k<bpp;++k) isum += pixels[id+k];
        mat(i,j) = isum/bpp*range + shift;
      }
    }
}

template<class T>
void mat2pixel(int width, int height, int bpp,   unsigned char * pixels, const T& mat, double range=1.0, unsigned char shift = 0){
//    mat.resize(height, width);
    for(int i=0;i<height;++i){
      for(int j=0;j<width;++j){
        int id = bpp*(j+width*i);
        char val = mat(i,j);
//        if( mat(i,j)>255.0) cout << "overflow" << endl;
 //       if( mat(i,j)<0.0) cout << "underflow" << endl;
        if(mat(i,j)>255.0) val = 255;
        if(mat(i,j)<0) val = 0;
        for(int k=0;k<bpp;++k)  pixels[id+k] = (unsigned char)val*range+shift;
      }
    }
}

void addspotnoize(int width, int height, int bpp, unsigned char * pixels, int x, int y, int size, unsigned char color){
    for(int i=y;i<y+size;++i){
      for(int j=x;j<x+size;++j){
        int id = bpp*(j+width*i);
        for(int k=0;k<bpp;++k)pixels[id+k] = color;
      }
    }
  
}

struct Abs : unary_function<val_t, val_t>{
  typedef val_t result_type;
  inline val_t operator ()(val_t v) const{
    return abs(v);
  }
};


class GoDec{
  public:
  GoDec(){}

/*  template <class Mat>
  GoDec(Mat& A){
    int r = (A.rows() < A.cols()) ? A.rows() : A.cols();
    run(A, r);
  }*/

  //template <class Mat>
  GoDec(const emat_t& X, const int r, const int k, const val_t eps ):L_t(X.rows(), X.cols()), S_t(X.rows(), X.cols()){
    run(X, r, k, eps);
  }

  void run(const emat_t& X, const int r, const int k, const val_t eps ){
    int loop= 0; val_t norm_rate = 0.0;
    val_t xnorm = X.norm();
    L_t = X;
    S_t; S_t.setZero(X.rows(), X.cols());// = Eigen::Zeros(X.rows(), X.cols());
    do{
    Eigen::JacobiSVD<emat_t> svd(X-S_t, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto sigmas = svd.singularValues();
    for(int i=r; i< sigmas.size();++i)sigmas[i]=0;
    L_t = svd.matrixU()*sigmas.asDiagonal()*svd.matrixV().transpose();

    S_t.setZero(X.rows(), X.cols());
    select_kbest(S_t, X-L_t, k, Abs());
    cout << "S norm: " << S_t.norm() << endl;
//    cout << S_t << endl;

    norm_rate = (X-L_t-S_t).norm()/xnorm;
    cout << loop++ << ": " << norm_rate << "=" << (X-L_t-S_t).norm() << "/" << xnorm <<   endl;
    
    }while(norm_rate > eps);
  }
  
  template<class F>
  inline void select_kbest(emat_t& s, const emat_t& m, int k, F f  ){
    vector<pair<int, val_t>> v;
    auto data = m.data();
    { int i=0;
    transform(data, data+m.size(), back_inserter(v), [&i](val_t datum){return make_pair(i++, datum);});
    }
    partial_sort(v.begin(), v.begin()+k, v.end(), [&f](const pair<int, val_t>& p1, const pair<int, val_t>& p2){return f(p1.second)>f(p2.second);});

    s.setZero(m.rows(), m.cols());
    for( auto it = v.begin(); it!=v.begin()+k; ++it){
      const auto& datum = *it;
      if(datum.second==0.0){
       cout << "called!!!" << endl;
      continue;
      }
      //Assume column first
      int col =  datum.first/m.rows();
      int row =  datum.first%m.rows();
//      cout << "(" << row << "," << col << ")" << endl;
      s(row, col) = datum.second;  
    }
  }
  
/*  template <class Mat>
  void run(Mat& A, const int rank){
    if (A.cols() == 0 || A.rows() == 0) return;
    int r = (rank < A.cols()) ? rank : A.cols();
    r = (r < A.rows()) ? r : A.rows();
    
    // Gaussian Random Matrix for A^T
    Eigen::MatrixXf O(A.rows(), r);
    sampleGaussianMat(O);
    
    // Compute Sample Matrix of A^T
    Eigen::MatrixXf Y = A.transpose() * O;
    
    // Orthonormalize Y
    processGramSchmidt(Y);

    // Range(B) = Range(A^T)
    Eigen::MatrixXf B = A * Y;
    
    // Gaussian Random Matrix
    Eigen::MatrixXf P(B.cols(), r);
    sampleGaussianMat(P);
    
    // Compute Sample Matrix of B
    Eigen::MatrixXf Z = B * P;
    
    // Orthonormalize Z
    processGramSchmidt(Z);
    
    // Range(C) = Range(B)
    Eigen::MatrixXf C = Z.transpose() * B; 
    
    Eigen::JacobiSVD<Eigen::MatrixXf> svdOfC(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    // C = USV^T
    // A = Z * U * S * V^T * Y^T()
    matU_ = Z * svdOfC.matrixU();
    matS_ = svdOfC.singularValues();
    matV_ = Y * svdOfC.matrixV();
  }
  
  const Eigen::MatrixXf& matrixU() const {
    return matU_;
  }*/

/*  const Eigen::VectorXf& singularValues() const {
    return matS_;
  }

  const Eigen::MatrixXf& matrixV() const {
    return matV_;
  }*/

  const emat_t& matrixL() const{
    return L_t;
  }
  const emat_t& matrixS() const{
    return S_t;
  }
private:
  emat_t L_t;
  emat_t S_t;
};

template <class T>
void histogram(const T& m){
  int prev= 0;
  for(int i=0;i<=255;++i){
    int ct = (m<=i).count();
    cout << "# of " << i << ": " << ct-prev <<  endl;
    prev=ct;
  }
  
}


void eigentest(){
   Eigen::Matrix<val_t, Eigen::Dynamic, Eigen::Dynamic,  Eigen::RowMajor> m (2,3);
   m << 0.3, 0.5, 0.1,
        0.6, 0.2, 0.4;
  auto data = m.data();
  for(int i=0;i<m.size();++i){
    cout << data[i] << endl;
  }
  
  vector<val_t> v;
  vector<pair<int, val_t>> v2;
  copy(data, data+6, back_inserter(v));
  { int i=0;
    transform(data, data+6, back_inserter(v2), [&i](val_t datum){return make_pair(i++, datum);});
  }
  partial_sort(v2.begin(), v2.begin()+3, v2.end(), [](const pair<int, val_t>& p1, const pair<int, val_t>& p2){return p1.second<p2.second;});
/*  for(auto val : v){
    cout << val << endl;
  }
  partial_sort(data, data+3, data+m.size());
  cout << m << endl;*/
  for(auto val : v2){
    cout << val.first << "," << val.second << endl;
  }


}


void godectest(){
    unsigned char* pixels ;
    int width;
    int height;
    int bpp;
    int ret;

    cout << "reading ...\n";
    pixels = stbi_load (FILE_NAME, &width, &height, &bpp, 0);
    if(pixels==NULL){
      cout << "Error: can't read file." << endl;
    }
    
    cout << "FILE_NAME = " << FILE_NAME     << "\n";
    cout << "pixels    = " << (void*)pixels << "\n";
    cout << "width     = " << width         << "\n";
    cout << "height    = " << height        << "\n";
    cout << "bpp       = " << bpp           << "\n";

    grayscale( width, height, bpp, pixels);//, 0.5, 64);
//    addspotnoize(width, height, bpp, pixels, 80, 60, 5, 255);
    emat_t  X;
    pixel2mat(width, height,  bpp,  pixels, X);
 //   Eigen::ArrayXXf a = m;
//    cout << (a>0).count() << endl;
//    histogram(a);

    int rank = (height   < width   ) ? height  : width;
    int rdiff = 2;
    int svdr = 10;
    //
    //To compare with usual SVD, k is set so that GoDec has the same memory usage with SVD.
    //width+height+1 = # of elem.s in 1 column in both U and V + 1 for singular value.
    int k    = rdiff*(width+height+1);
    int godecr = svdr - rdiff;

//    GoDec godec(X, godecr, k, 0.378);
    GoDec godec(X, godecr, k, 0.1294);
    emat_t L = godec.matrixL();

    unsigned char pixelsL[height*width*bpp];
    for(int i=0;i<height*width*bpp;++i)pixelsL[i]=0;
    mat2pixel(width, height,  bpp,  pixelsL, L, 0.5, 64);
    cout << "||X-L||= " << (X-L).norm() << endl;
    cout << "writing L...\n";
    ret = stbi_write_png (L_FILE_NAME, width, height, bpp, pixelsL, width*bpp);

    cout << "Succeeded in writing?: " << ret << "\n";

    emat_t S = godec.matrixS();
    unsigned char pixelsS[height*width*bpp];
    for(int i=0;i<height*width*bpp;++i)pixelsS[i]=0;
    mat2pixel(width, height,  bpp,  pixelsS, S, 0.5, 64);
    cout << "||X-S||= " << (X-S).norm() << endl;
    cout << "writing S...\n";
    ret = stbi_write_png (S_FILE_NAME, width, height, bpp, pixelsS, width*bpp);
    cout << "Succeeded in writing?: " << ret << "\n";


//    emat_t LpS = L + S;
    unsigned char pixelsLpS[height*width*bpp];
    for(int i=0;i<height*width*bpp;++i)pixelsLpS[i]=0;
    mat2pixel(width, height,  bpp,  pixelsLpS, L+S, 0.5, 64);
    cout << "||X-LpS||= " << (X-(L+S)).norm() << endl;
    cout << "writing LpS...\n";
    ret = stbi_write_png (LpS_FILE_NAME, width, height, bpp, pixelsLpS, width*bpp);
    cout << "Succeeded in writing?: " << ret << "\n";


    Eigen::JacobiSVD<emat_t> svd(
      X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto sigmas =  svd.singularValues();
//    cout << "rank: " << sigmas.size() << endl;*/
    for(int k=svdr; k< sigmas.size();++k)sigmas[k]=0;
    emat_t compared  = svd.matrixU()*sigmas.asDiagonal()*svd.matrixV().transpose();
//    cout << "Simple r-rank SVD achives: ||X-X~||=" << (X-compared).norm() << endl;

    unsigned char pixelssvd[height*width*bpp];
    for(int i=0;i<height*width*bpp;++i)pixelssvd[i]=0;
    mat2pixel(width, height,  bpp,  pixelssvd, compared, 0.5, 64);
    cout << "||X-svd||= " << (X-compared).norm() << endl;
    cout << "writing svd...\n";
    ret = stbi_write_png (SVD_FILE_NAME, width, height, bpp, pixelssvd, width*bpp);
    cout << "Succeeded in writing?: " << ret << "\n";
    

    unsigned char pixelsX[height*width*bpp];
    for(int i=0;i<height*width*bpp;++i)pixelsX[i]=0;
    mat2pixel(width, height,  bpp,  pixelsX, X , 0.5, 64);
    cout << "writing X...\n";
    ret = stbi_write_png (X_FILE_NAME, width, height, bpp, pixelsX, width*bpp);
    cout << "Succeeded in writing?: " << ret << "\n";

    stbi_image_free (pixels);
}


/**
 *
 */
int main (int argc, char** argv) 
{

//    eigentest();
  godectest();


    return 0;
}
