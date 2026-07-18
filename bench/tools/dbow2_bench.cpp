// DBoW2 retrieval benchmark against OUR reloc protocol's probe frames.
// Database = every STRIDE-th frame of the pack (ORB features -> BoW);
// queries = the exact probe frame indices our rl19 wake-up runs used.
// Correctness: a retrieved keyframe whose position (from the fz19 map
// trajectory, time-interpolated) lies within R metres of the probe
// frame's position. Reports recall@1 / recall@5 at 3 m and 5 m.
//
// Build (on .58): see dbow2_build.sh. Usage:
//   dbow2_bench <voc.txt> <pack_dir> <W> <H> <ref_map.tum> <probes.txt>
#include "DBoW2/TemplatedVocabulary.h"
#include "DBoW2/FORB.h"
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
    ORBVocabulary;

static std::vector<cv::Mat> toVec(const cv::Mat& d) {
  std::vector<cv::Mat> v;
  v.reserve(d.rows);
  for (int i = 0; i < d.rows; i++) v.push_back(d.row(i));
  return v;
}

int main(int argc, char** argv) {
  if (argc < 7) { fprintf(stderr, "args\n"); return 1; }
  const char* vocp = argv[1];
  std::string pack = argv[2];
  int W = atoi(argv[3]), H = atoi(argv[4]);
  const char* refp = argv[5];
  const char* probesp = argv[6];
  const int STRIDE = 12;
  const double EXCL_S = 3.0;           /* temporal self-match exclusion */
  const int DB_CAP = 400;              /* match our map's keyframe cap */

  // ref trajectory: ts -> pos
  std::vector<double> rt; std::vector<std::array<double,3>> rp;
  { FILE* f = fopen(refp, "r"); if (!f) { perror("ref"); return 1; }
    double t,x,y,z,qx,qy,qz,qw;
    while (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf", &t,&x,&y,&z,&qx,&qy,&qz,&qw) == 8) {
      rt.push_back(t); rp.push_back({x,y,z});
    } fclose(f); }
  auto posAt = [&](double t, std::array<double,3>& out)->bool {
    auto it = std::lower_bound(rt.begin(), rt.end(), t);
    size_t i = it - rt.begin();
    if (i >= rt.size()) i = rt.size()-1;
    if (i > 0 && fabs(rt[i-1]-t) < fabs(rt[i]-t)) i--;
    if (fabs(rt[i]-t) > 0.15) return false;
    out = rp[i]; return true;
  };

  // probe frame indices
  std::vector<long> probes;
  { FILE* f = fopen(probesp, "r"); long v;
    while (fscanf(f, "%ld", &v) == 1) probes.push_back(v); fclose(f); }

  // frames.csv timestamps
  std::vector<uint64_t> fts;
  { FILE* f = fopen((pack+"/frames.csv").c_str(), "r");
    if (!f) { perror("frames.csv"); return 1; }
    uint64_t ts; long idx;
    while (fscanf(f, "%llu,%ld\n", (unsigned long long*)&ts, &idx) == 2)
      fts.push_back(ts);
    fclose(f); }
  fprintf(stderr, "frames=%zu probes=%zu\n", fts.size(), probes.size());

  ORBVocabulary voc;
  fprintf(stderr, "loading voc...\n");
  voc.loadFromTextFile(vocp);
  fprintf(stderr, "voc ok\n");

  cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
  FILE* raw = fopen((pack+"/frames.raw").c_str(), "rb");
  if (!raw) { perror("frames.raw"); return 1; }
  std::vector<uint8_t> buf((size_t)2*W*H);

  struct KF { DBoW2::BowVector bow; double t; std::array<double,3> p; };
  std::vector<KF> kfs;
  std::map<long, DBoW2::BowVector> probeBow;
  std::map<long, std::array<double,3>> probePos;

  for (long i = 0; i < (long)fts.size(); i++) {
    bool isKf = (i % STRIDE) == 0;
    bool isProbe = std::find(probes.begin(), probes.end(), i) != probes.end();
    if (!isKf && !isProbe) { fseek(raw, (long)2*W*H, SEEK_CUR); continue; }
    if (fread(buf.data(), (size_t)2*W*H, 1, raw) != 1) break;
    cv::Mat img(H, W, CV_8UC1, buf.data());   // left eye
    std::vector<cv::KeyPoint> kp; cv::Mat desc;
    orb->detectAndCompute(img, cv::noArray(), kp, desc);
    if (desc.rows < 20) continue;
    DBoW2::BowVector bv; DBoW2::FeatureVector fv;
    voc.transform(toVec(desc), bv, fv, 4);
    double t = fts[i] * 1e-9;
    std::array<double,3> p;
    if (isKf && posAt(t, p)) kfs.push_back({bv, t, p});
    if (isProbe) { probeBow[i] = bv;
      if (posAt(t, p)) probePos[i] = p; }
  }
  fclose(raw);
  fprintf(stderr, "db kfs=%zu\n", kfs.size());

  int n = 0, r1_3 = 0, r5_3 = 0, r1_5 = 0, r5_5 = 0;
  for (long pi : probes) {
    if (!probeBow.count(pi) || !probePos.count(pi)) continue;
    auto& q = probeBow[pi]; auto& qp = probePos[pi];
    double qt = fts[pi] * 1e-9;
    std::vector<std::pair<double,int>> sc;
    int step = kfs.size() > (size_t)DB_CAP ? (int)(kfs.size() / DB_CAP) : 1;
    for (int k = 0; k < (int)kfs.size(); k += step) {
      if (fabs(kfs[k].t - qt) < EXCL_S) continue;   /* place, not moment */
      sc.push_back({voc.score(q, kfs[k].bow), k});
    }
    if (sc.size() < 5) continue;
    std::sort(sc.rbegin(), sc.rend());
    auto dist = [&](int k) {
      double dx = kfs[k].p[0]-qp[0], dy = kfs[k].p[1]-qp[1], dz = kfs[k].p[2]-qp[2];
      return sqrt(dx*dx+dy*dy+dz*dz); };
    double d1 = dist(sc[0].second);
    double best5 = d1;
    for (int j = 1; j < 5 && j < (int)sc.size(); j++)
      best5 = std::min(best5, dist(sc[j].second));
    n++;
    if (d1 <= 3.0) r1_3++;
    if (best5 <= 3.0) r5_3++;
    if (d1 <= 5.0) r1_5++;
    if (best5 <= 5.0) r5_5++;
    printf("PROBE frame=%ld top1_d=%.2f best5_d=%.2f top1_score=%.4f\n",
           pi, d1, best5, sc[0].first);
    printf("CAND frame=%ld", pi);
    for (int j = 0; j < 5 && j < (int)sc.size(); j++)
      printf(" %.9f", kfs[sc[j].second].t);
    printf("\n");
  }
  printf("DBOW2-SUMMARY n=%d r1@3m=%.3f r5@3m=%.3f r1@5m=%.3f r5@5m=%.3f\n",
         n, n?(double)r1_3/n:0, n?(double)r5_3/n:0, n?(double)r1_5/n:0,
         n?(double)r5_5/n:0);
  return 0;
}
