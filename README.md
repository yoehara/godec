#Nai:ve GoDec Implementation

Implementaiton of the following paper (referred as Nai:ve GoDec)
GoDec: Randomized Low-rank & Sparse Matrix Decomposition in Noisy Case, Tianyi Zhou, Dacheng Tao, ICML2011
http://www.icml-2011.org/papers/41_icmlpaper.pdf

##Build
Uses c++0x features. May require gcc 4.6.0 or higher to build.
```
git clone git@github.com:niam/godec.git
cd godec
./waf configure
./waf build
```

##Usage
```
build/src/godec -r 10 -k 100 hogehoge.png
```
This creates L.png, S.png, X.png, and LpS.png.
 - X.png: original image automatically grayscaled.
 - L.png: L in GoDec.
 - S.png: S in GoDec.
 - LpS.png: L+S in GoDec.

##License
I used stb_image.c and stb_image_write.c, an image library claimed to be public domain.
See http://nothings.org/ for details.
Everything else other than this library is under New BSD License.
I implemented for research use. I do NOT know patent issues on GoDec. I have never investigated them. Beaware with patent issues for commercial use.

##References
 - http://www.icml-2011.org/papers/41_icmlpaper.pdf
 - http://nothings.org/

