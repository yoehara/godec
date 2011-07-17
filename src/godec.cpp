/* 
 *  Copyright (c) 2011 Yo Ehara
 * 
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 * 
 *   1. Redistributions of source code must retain the above Copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above Copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the name of the authors nor the names of its contributors
 *      may be used to endorse or promote products derived from this
 *      software without specific prior written permission.
 */

#include <iostream>
#include "stbi/stb_image.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stbi/stb_image_write.c"
#include <eigen3/Eigen/Dense>
#include "cmdline.h"
typedef float  val_t;
using namespace std;

const char* FILE_NAME=NULL;
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
void mat2pixel(int width, int height, int bpp,   unsigned char * pixels, const T& mat,  double range=1.0, unsigned char shift = 0, bool isclip=true){
//    mat.resize(height, width);
    for(int i=0;i<height;++i){
      for(int j=0;j<width;++j){
        int id = bpp*(j+width*i);
        char val = mat(i,j);
//        if( mat(i,j)>255.0) cout << "overflow" << endl;
 //       if( mat(i,j)<0.0) cout << "underflow" << endl;
        if(isclip){
          if(mat(i,j)>255.0) val = 255;
          if(mat(i,j)<0) val = 0;
        }
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


class godec_t{
  public:
  godec_t(){}


  godec_t(const emat_t& X, const int r, const int k, const val_t eps2 ):L_t(X.rows(), X.cols()), S_t(X.rows(), X.cols()){
    run(X, r, k, eps2);
  }

  void run(const emat_t& X, const int r, const int k, const val_t eps2 , bool autoconverge=true ){
    int loop= 0; val_t norm_rate = 0.0; val_t prevnorm = -1.0; val_t nownorm= 0.0;
    val_t xnorm = X.norm();
    L_t = X;
    S_t; S_t.setZero(X.rows(), X.cols());
    do{
      Eigen::JacobiSVD<emat_t> svd(X-S_t, Eigen::ComputeThinU | Eigen::ComputeThinV);
      auto sigmas = svd.singularValues();
      for(int i=r; i< sigmas.size();++i)sigmas[i]=0;
      L_t = svd.matrixU()*sigmas.asDiagonal()*svd.matrixV().transpose();

      S_t.setZero(X.rows(), X.cols());
      select_kbest(S_t, X-L_t, k, Abs());
      cout << "S norm: " << S_t.norm() << endl;

      prevnorm = nownorm;
      nownorm   = (X-L_t-S_t).norm();
      norm_rate = nownorm/xnorm;
      cout << loop++ << ": " << norm_rate << "=" << nownorm << "/" << xnorm <<   endl;
      cout << prevnorm << ", " << nownorm << " " << (prevnorm-nownorm) << endl;
      
    }while( !(autoconverge && abs(prevnorm-nownorm)<eps2));
  //norm_rate > eps &&
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



void godectest(int svdr, int k, val_t eps2, bool svdcomparemode,  int rdiff, float range, int shift , bool isclip){
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

    cout << "rank      = " << svdr          << "\n";
    cout << "card      = " << k             << "\n";
    cout << "eps       = " << eps2          << "\n";
    cout << "comparesvd= " << svdcomparemode<< "\n";
    cout << "rdiff     = " << rdiff         << "\n";
    cout << "range     = " << range         << "\n";
    cout << "shift     = " << shift         << "\n";
    cout << "isclip    = " << isclip        << "\n";

    grayscale( width, height, bpp, pixels);//, 0.5, 64);
//    addspotnoize(width, height, bpp, pixels, 80, 60, 5, 255);
    emat_t  X;
    pixel2mat(width, height,  bpp,  pixels, X);

    int rank = (height   < width   ) ? height  : width;
    int godecr = -1;
    if(svdcomparemode){
    //To compare with usual SVD, k is set so that godec_t has the same memory usage with SVD.
    //width+height+1 = # of elem.s in 1 column in both U and V + 1 for singular value.
       k    = rdiff*(width+height+1);
       godecr = svdr - rdiff;
    }else{
      godecr = svdr;
    }

    godec_t godec(X, godecr, k, eps2);
    emat_t L = godec.matrixL();

    unsigned char pixelsL[height*width*bpp];
    for(int i=0;i<height*width*bpp;++i)pixelsL[i]=0;
    mat2pixel(width, height,  bpp,  pixelsL, L, range, shift, isclip);
    cout << "||X-L||= " << (X-L).norm() << endl;
    cout << "writing L...\n";
    ret = stbi_write_png (L_FILE_NAME, width, height, bpp, pixelsL, width*bpp);
    cout << "Succeeded in writing?: " << ret << "\n";

    emat_t S = godec.matrixS();
    unsigned char pixelsS[height*width*bpp];
    for(int i=0;i<height*width*bpp;++i)pixelsS[i]=0;
    mat2pixel(width, height,  bpp,  pixelsS, S, range, shift, isclip);
    cout << "||X-S||= " << (X-S).norm() << endl;
    cout << "writing S...\n";
    ret = stbi_write_png (S_FILE_NAME, width, height, bpp, pixelsS, width*bpp);
    cout << "Succeeded in writing?: " << ret << "\n";


    unsigned char pixelsLpS[height*width*bpp];
    for(int i=0;i<height*width*bpp;++i)pixelsLpS[i]=0;
    mat2pixel(width, height,  bpp,  pixelsLpS, L+S, range, shift, isclip);
    cout << "||X-LpS||= " << (X-(L+S)).norm() << endl;
    cout << "writing LpS...\n";
    ret = stbi_write_png (LpS_FILE_NAME, width, height, bpp, pixelsLpS, width*bpp);
    cout << "Succeeded in writing?: " << ret << "\n";


    if(svdcomparemode){
      Eigen::JacobiSVD<emat_t> svd(
        X, Eigen::ComputeThinU | Eigen::ComputeThinV);
      auto sigmas =  svd.singularValues();
      for(int k=svdr; k< sigmas.size();++k)sigmas[k]=0;
      emat_t compared  = svd.matrixU()*sigmas.asDiagonal()*svd.matrixV().transpose();
//    cout << "Simple r-rank SVD achives: ||X-X~||=" << (X-compared).norm() << endl;

      unsigned char pixelssvd[height*width*bpp];
      for(int i=0;i<height*width*bpp;++i)pixelssvd[i]=0;
      mat2pixel(width, height,  bpp,  pixelssvd, compared, range, shift, isclip);
      cout << "||X-svd||= " << (X-compared).norm() << endl;
      cout << "writing svd...\n";
      ret = stbi_write_png (SVD_FILE_NAME, width, height, bpp, pixelssvd, width*bpp);
      cout << "Succeeded in writing?: " << ret << "\n";
     }
    

    unsigned char pixelsX[height*width*bpp];
    for(int i=0;i<height*width*bpp;++i)pixelsX[i]=0;
    mat2pixel(width, height,  bpp,  pixelsX, X , range, shift, isclip);
    cout << "writing X...\n";
    ret = stbi_write_png (X_FILE_NAME, width, height, bpp, pixelsX, width*bpp);
    cout << "Succeeded in writing?: " << ret << "\n";

    stbi_image_free (pixels);
}


int main (int argc, char** argv) 
{
  cmdline::parser a;
  a.add<int>("rank", 'r', "rank r.", false, 10);
  a.add<int>("cardinality", 'k', "cardinality k.", false, 1000);
  a.add<float>("eps", 'e', "iteration terminates until |previous error - current error|<eps", false, 0.1);
  a.add<bool>("comparesvdmode", 0, "comparison with svd in the same degree of freedom.", false, false);
  a.add<int>("rdiff", 0, "# of ranks used for cardinality k in comparesvdmode. The cardinality k is automatically tuned to rdiff*(height+width+1) so that it has the same degree of freedom with svd.", false, 2);
  a.add<float>("range", 0, "output changed to shift ... 255*range + shift color", false, 1.0);
  a.add<int>("shift", 0, "output changed to shift ... 255*range + shift color", false, 0);
  a.add<bool>("isclip", 0, ">255, <0 values in a matrix are clipped to 255, 0, respectively.", false, true);
  a.add("help", 0, "print this message");
  a.footer("filename ...");
  bool ok=a.parse(argc, argv);

  if (argc==1 ||  a.exist("help")){
    cerr<<a.usage();
    return 0;
  }
  
  if (!ok){
    cerr<<a.error()<<endl<<a.usage();
    return 0;
  }
  
  FILE_NAME = a.rest()[0].c_str();

    godectest(a.get<int>("rank"), a.get<int>("cardinality"), a.get<float>("eps"),a.get<bool>("comparesvdmode"), a.get<int>("rdiff"), a.get<float>("range"), a.get<int>("shift"), a.get<bool>("isclip") );

    return 0;
}
