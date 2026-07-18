#!/bin/bash
# Build DBoW2 (from ORB-SLAM3's tree) + the retrieval benchmark, then
# run it on the reloc-grid sequences reusing the rl19 probe frames.
set -e
cd /root/orbslam3_monado/Thirdparty/DBoW2
[ -f lib/libDBoW2.so ] || (mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null && make -j8 > /dev/null)
cd /root/orbslam3_monado/Vocabulary
[ -f ORBvoc.txt ] || tar xzf ORBvoc.txt.tar.gz
g++ -O2 -std=c++14 /root/dbow2_bench.cpp -o /root/dbow2_bench \
  -I/root/orbslam3_monado/Thirdparty/DBoW2 \
  -I/usr/include/opencv4 \
  -L/root/orbslam3_monado/Thirdparty/DBoW2/lib -lDBoW2 \
  -lopencv_core -lopencv_features2d -lopencv_imgproc \
  -Wl,-rpath,/root/orbslam3_monado/Thirdparty/DBoW2/lib
echo BUILD-OK
O=/mnt/processing/dbow2_bench
mkdir -p $O
for s in dataset-corridor1_512_16 dataset-corridor2_512_16 dataset-corridor3_512_16 dataset-corridor4_512_16 dataset-corridor5_512_16 dataset-magistrale2_512_16 dataset-room1_512_16; do
  L=/mnt/processing/relocgrid19/${s}__rl19.log
  [ -f "$L" ] || continue
  grep -o 'RELOC k=[0-9]* frame=[0-9]*' $L | grep -o 'frame=[0-9]*' | cut -d= -f2 > $O/${s}.probes
  /root/dbow2_bench /root/orbslam3_monado/Vocabulary/ORBvoc.txt \
    /mnt/processing/packs/$s 512 512 \
    /mnt/processing/relocgrid19/${s}__rl19_map.tum \
    $O/${s}.probes > $O/${s}.out 2> $O/${s}.err
  grep DBOW2-SUMMARY $O/${s}.out | sed "s/^/$s /"
done
echo ALL-DONE
