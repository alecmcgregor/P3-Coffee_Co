#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include "DecisionTree.h"

// helpers
double calcRmse(const std::vector<double>& y, const std::vector<double>& p) {
    double sq = 0;
    for (size_t i = 0; i < y.size(); ++i)
        sq += (y[i] - p[i])*(y[i] - p[i]);
    return std::sqrt(sq / y.size());
}
double calcR2(const std::vector<double>& y, const std::vector<double>& p) {
    double mean = std::accumulate(y.begin(), y.end(), 0.0)/y.size();
    double res = 0, tot = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        res += (y[i] - p[i])*(y[i] - p[i]);
        tot += (y[i] - mean)*(y[i] - mean);
    }
    return 1 - res/tot;
}

int main() {
    const std::string path =
      "C:\\Users\\Josh\\ClionProjects\\untitled1\\generated_coffee.csv";
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << path << "\n";
        return 1;
    }

    // header→cols
    std::string header; std::getline(file, header);
    std::vector<std::string> cols;
    { std::stringstream ss(header);
      std::string c;
      while (std::getline(ss, c, ',')) cols.push_back(c);
    }

    auto idx_of = [&](const std::string& name){
      auto it = std::find(cols.begin(), cols.end(), name);
      return it==cols.end()? -1 : int(std::distance(cols.begin(), it));
    };

    // features & target
    std::vector<std::string> feats = {
      "Data.Scores.Aroma","Data.Scores.Aftertaste","Data.Scores.Acidity",
      "Data.Scores.Body","Data.Scores.Balance","Data.Scores.Uniformity",
      "Data.Scores.Sweetness","Data.Scores.Moisture"
    };
    std::vector<int> fidx;
    for (auto& f: feats) {
      int i = idx_of(f);
      if (i<0) { std::cerr<<"Missing column "<<f<<"\n"; return 1; }
      fidx.push_back(i);
    }
    int tidx = idx_of("Data.Scores.Flavor");
    if (tidx<0) { std::cerr<<"Missing column Data.Scores.Flavor\n"; return 1; }

    // load rows
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::string line;
    while (std::getline(file, line)) {
      std::stringstream ls(line);
      std::vector<std::string> row;
      std::string cell;
      while (std::getline(ls, cell, ',')) row.push_back(cell);
      if (row.size()<=size_t(tidx)) continue;
      bool bad=false;
      std::vector<double> xv;
      for (int i: fidx) {
        if (row[i].empty()){ bad=true; break; }
        xv.push_back(std::stod(row[i]));
      }
      if (bad||row[tidx].empty()) continue;
      X.push_back(xv);
      y.push_back(std::stod(row[tidx]));
    }

    // split
    int n = int(X.size());
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937{std::random_device{}()});
    int tr = int(0.8*n);
    std::vector<std::vector<double>> Xtr, Xte;
    std::vector<double> ytr, yte;
    for (int i=0;i<n;++i){
      if (i<tr) { Xtr.push_back(X[idx[i]]); ytr.push_back(y[idx[i]]); }
      else      { Xte.push_back(X[idx[i]]); yte.push_back(y[idx[i]]); }
    }

    // hyperparams
    int max_depth = 12, min_samples_split = 20;
    std::cout<<"Params: depth="<<max_depth
             <<", min_split="<<min_samples_split<<"\n";

    // train
    DecisionTree model(max_depth, min_samples_split);
    model.fit(Xtr, ytr);

    // train metrics
    auto tr_pred = model.predict(Xtr);
    std::cout<<"Train RMSE="<<calcRmse(ytr,tr_pred)
             <<", R2="<<calcR2(ytr,tr_pred)<<"\n";

    // test metrics
    auto te_pred = model.predict(Xte);
    std::cout<<"Test  RMSE="<<calcRmse(yte,te_pred)
             <<", R2="<<calcR2(yte,te_pred)<<"\n";

    // importances
    std::cout<<"\nFeature importances:\n";
    auto imps = model.feature_importances();
    for (size_t i=0;i<feats.size();++i)
        std::cout<<feats[i]<<": "<<imps[i]<<"\n";

    return 0;
}

