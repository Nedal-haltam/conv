.PHONY: all build run-all-images run-all-videos all-vids all-imgs clean c-ffi c-dll

LIB_OPENCV := $(shell pkg-config --cflags --libs opencv4)
LIBS := $(LIB_OPENCV) -L./fftw-3.3.10/fftw-3.3.10/.libs/ -l:libfftw3.a -lm

build: main.cpp
	g++ main.cpp -O3 -march=native -o ./build/main $(LIBS)

run-all-images:
# 	./build/main -i ./input_images/lena.png -o ./output_images/lena.png
# 	./build/main -i ./input_images/tree.png -o ./output_images/tree.png
# 	./build/main -i ./input_images/humananflower.png -o ./output_images/humananflower.png
	./build/main -i ./input_images/lena.ppm -o ./output_images/lena.ppm
	./build/main -i ./input_images/tree.ppm -o ./output_images/tree.ppm
	./build/main -i ./input_images/humananflower.ppm -o ./output_images/humananflower.ppm

c-ffi:
	g++ -shared -fPIC -O3 -march=native -o ./build/libconv3d.so conv3d.cpp $(LIB_OPENCV)
c-dll:
	g++ -o ./build/conv3d.dll conv3d.cpp -shared -O3 -march=native -Wl,--export-all-symbols

clean:
	rm -rf ./build/
	mkdir -p ./build/
	rm -rf ./output_images/
	mkdir -p ./output_images/
	rm -rf ./output_videos/
	mkdir -p ./output_videos/
