/**
 * Native Cantera C++ BandMatrix Oracle Runner.
 * Evaluates corpus_band_matrix.jsonl against pinned Cantera BandMatrix.
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include <cantera/numerics/BandMatrix.h>
#include <cantera/base/ctexceptions.h>

using namespace Cantera;

// Minimal JSON line parser for corpus entries
struct MatrixEntry {
    size_t i;
    size_t j;
    double val;
};

void run_cases(const std::string& corpus_path, const std::string& output_path, const std::string& run_id) {
    std::ifstream in(corpus_path);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << corpus_path << std::endl;
        exit(1);
    }

    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << output_path << std::endl;
        exit(1);
    }

    out << "[\n";
    std::string line;
    bool first_case = true;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        // Parse JSON fields simply
        auto get_str = [&](const std::string& key) -> std::string {
            size_t p = line.find("\"" + key + "\":");
            if (p == std::string::npos) return "";
            size_t q1 = line.find("\"", p + key.length() + 3);
            size_t q2 = line.find("\"", q1 + 1);
            return line.substr(q1 + 1, q2 - q1 - 1);
        };
        auto get_int = [&](const std::string& key, int def = 0) -> int {
            size_t p = line.find("\"" + key + "\":");
            if (p == std::string::npos) return def;
            size_t start = p + key.length() + 3;
            size_t end = line.find_first_of(",}", start);
            return std::stoi(line.substr(start, end - start));
        };
        auto get_bool = [&](const std::string& key, bool def = false) -> bool {
            size_t p = line.find("\"" + key + "\":");
            if (p == std::string::npos) return def;
            return line.find("true", p) < line.find("false", p) || (line.find("true", p) != std::string::npos && line.find("false", p) == std::string::npos);
        };

        std::string case_id = get_str("case_id");
        std::string category = get_str("category");
        size_t n = get_int("n");
        size_t kl = get_int("kl");
        size_t ku = get_int("ku");
        size_t nrhs = get_int("nrhs", 1);
        bool is_singular = get_bool("is_singular");

        // Parse entries: "entries": [[i, j, val], ...]
        std::vector<MatrixEntry> entries;
        size_t ent_pos = line.find("\"entries\":");
        if (ent_pos != std::string::npos && n > 0) {
            size_t b_open = line.find("[", ent_pos + 9);
            size_t b_close = line.find("]],", b_open);
            std::string sub = line.substr(b_open + 1, b_close - b_open);
            size_t cur = 0;
            while ((cur = sub.find("[", cur)) != std::string::npos) {
                size_t c_end = sub.find("]", cur);
                std::string item = sub.substr(cur + 1, c_end - cur - 1);
                std::stringstream ss(item);
                std::string token;
                std::vector<std::string> parts;
                while (std::getline(ss, token, ',')) {
                    parts.push_back(token);
                }
                if (parts.size() >= 3) {
                    MatrixEntry me;
                    me.i = std::stoul(parts[0]);
                    me.j = std::stoul(parts[1]);
                    me.val = std::stod(parts[2]);
                    entries.push_back(me);
                }
                cur = c_end + 1;
            }
        }

        // Parse rhs: "rhs": [[col0, col1, ...], ...]
        std::vector<double> rhs_flat(n * nrhs, 0.0);
        size_t rhs_pos = line.find("\"rhs\":");
        if (rhs_pos != std::string::npos && n > 0 && nrhs > 0) {
            size_t b_open = line.find("[", rhs_pos + 5);
            size_t b_close = line.find("]],", b_open);
            if (b_close == std::string::npos) b_close = line.find("]]}", b_open);
            std::string sub = line.substr(b_open + 1, b_close - b_open);
            size_t cur = 0;
            size_t row_idx = 0;
            while ((cur = sub.find("[", cur)) != std::string::npos && row_idx < n) {
                size_t c_end = sub.find("]", cur);
                std::string item = sub.substr(cur + 1, c_end - cur - 1);
                std::stringstream ss(item);
                std::string token;
                size_t col_idx = 0;
                while (std::getline(ss, token, ',') && col_idx < nrhs) {
                    // column-major flat array for solve(rhs, nrhs, n)
                    rhs_flat[col_idx * n + row_idx] = std::stod(token);
                    col_idx++;
                }
                row_idx++;
                cur = c_end + 1;
            }
        }

        if (!first_case) out << ",\n";
        first_case = false;

        out << std::setprecision(17);
        out << "  {\n";
        out << "    \"case_id\": \"" << case_id << "\",\n";
        out << "    \"category\": \"" << category << "\",\n";
        out << "    \"run_id\": \"" << run_id << "\",\n";

        if (n == 0) {
            BandMatrix m(0, 0, 0);
            out << "    \"status\": \"PASS\",\n";
            out << "    \"nRows\": " << m.nRows() << ",\n";
            out << "    \"nColumns\": " << m.nColumns() << ",\n";
            out << "    \"nSubDiagonals\": " << m.nSubDiagonals() << ",\n";
            out << "    \"nSuperDiagonals\": " << m.nSuperDiagonals() << ",\n";
            out << "    \"ldim\": " << m.ldim() << ",\n";
            out << "    \"oneNorm\": " << m.oneNorm() << ",\n";
            out << "    \"solution\": []\n";
            out << "  }";
            continue;
        }

        BandMatrix matrix(n, kl, ku);
        for (const auto& e : entries) {
            matrix(e.i, e.j) = e.val;
        }

        out << "    \"nRows\": " << matrix.nRows() << ",\n";
        out << "    \"nColumns\": " << matrix.nColumns() << ",\n";
        out << "    \"nSubDiagonals\": " << matrix.nSubDiagonals() << ",\n";
        out << "    \"nSuperDiagonals\": " << matrix.nSuperDiagonals() << ",\n";
        out << "    \"ldim\": " << matrix.ldim() << ",\n";
        out << "    \"factorAlgorithm\": " << matrix.factorAlgorithm() << ",\n";

        double one_norm = matrix.oneNorm();
        out << "    \"oneNorm\": " << one_norm << ",\n";

        double val_row_small = 0.0;
        size_t i_small = matrix.checkRows(val_row_small);
        out << "    \"checkRows_iSmall\": " << i_small << ",\n";
        out << "    \"checkRows_valSmall\": " << val_row_small << ",\n";

        double val_col_small = 0.0;
        size_t j_small = matrix.checkColumns(val_col_small);
        out << "    \"checkColumns_jSmall\": " << j_small << ",\n";
        out << "    \"checkColumns_valSmall\": " << val_col_small << ",\n";

        // Mult test with column 0 of rhs
        std::vector<double> b0(n);
        for (size_t i = 0; i < n; i++) b0[i] = rhs_flat[i];
        std::vector<double> mult_res(n, 0.0);
        matrix.mult(b0, mult_res);

        out << "    \"mult_result\": [";
        for (size_t i = 0; i < n; i++) {
            if (i > 0) out << ", ";
            out << mult_res[i];
        }
        out << "],\n";

        std::vector<double> left_mult_res(n, 0.0);
        matrix.leftMult(b0, left_mult_res);
        out << "    \"leftMult_result\": [";
        for (size_t i = 0; i < n; i++) {
            if (i > 0) out << ", ";
            out << left_mult_res[i];
        }
        out << "],\n";

        // Attempt factor & solve
        try {
            matrix.factor();
            out << "    \"status\": \"PASS\",\n";
            out << "    \"factor_info\": " << matrix.info() << ",\n";

            double rc = matrix.rcond(one_norm);
            out << "    \"rcond\": " << rc << ",\n";

            std::vector<double> sol = rhs_flat;
            matrix.solve(sol, nrhs, n);

            out << "    \"solution\": [";
            for (size_t i = 0; i < sol.size(); i++) {
                if (i > 0) out << ", ";
                out << sol[i];
            }
            out << "]\n";
        } catch (const CanteraError& err) {
            out << "    \"status\": \"SINGULAR\",\n";
            out << "    \"error_message\": \"" << err.getMessage() << "\",\n";
            out << "    \"factor_info\": " << matrix.info() << ",\n";
            out << "    \"solution\": []\n";
        }

        out << "  }";
    }
    out << "\n]\n";
    std::cout << "Native oracle completed -> " << output_path << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: run_native_band_matrix_oracle <corpus_path> <output_path> <run_id>\n";
        return 1;
    }
    run_cases(argv[1], argv[2], argv[3]);
    return 0;
}
