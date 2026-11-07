/**
 * Pinned native Cantera Domain1D differential validation harness.
 *
 * Compiles against native Cantera in WSL2 and executes the deterministic corpus
 * in audit/corpus_domain1d.jsonl, writing exact oracle outputs to
 * audit/native_domain1d_oracle_results.json.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>

#include "cantera/oneD/Domain1D.h"
#include "cantera/oneD/OneDim.h"
#include "cantera/oneD/refine.h"
#include "cantera/base/AnyMap.h"
#include "cantera/base/ctexceptions.h"

using namespace Cantera;
using namespace std;

class TestDomain1D : public Domain1D
{
public:
    explicit TestDomain1D(size_t nv=1, size_t points=1, double time=0.0)
        : Domain1D(nv, points, time)
    {
    }

    using Domain1D::getMeta;
    using Domain1D::setMeta;
    using Domain1D::locate;

    double initialValue(size_t n, size_t j) override
    {
        return 100.0 * (n + 1) + 1.0 * (j + 1);
    }

    void eval(size_t j, span<const double> x, span<double> r, span<int> mask, double rdt=0.0) override
    {
    }
};

int main(int argc, char** argv)
{
    string corpus_path = "audit/corpus_domain1d.jsonl";
    string output_path = "audit/native_domain1d_oracle_results.json";

    if (argc > 1) {
        corpus_path = argv[1];
    }
    if (argc > 2) {
        output_path = argv[2];
    }

    ifstream infile(corpus_path);
    if (!infile.is_open()) {
        cerr << "Failed to open corpus file: " << corpus_path << endl;
        return 1;
    }

    vector<AnyMap> results;
    string line;
    size_t case_idx = 0;

    while (getline(infile, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        AnyMap case_data = AnyMap::fromYamlString(line);
        string case_id = case_data["case_id"].asString();

        AnyMap out_rec;
        out_rec["case_id"] = case_id;
        out_rec["index"] = static_cast<long int>(case_idx);
        out_rec["success"] = true;
        out_rec["exception_thrown"] = false;
        out_rec["error_message"] = "";

        try {
            size_t nv = case_data.hasKey("nv") ? static_cast<size_t>(case_data["nv"].as<long int>()) : 1;
            size_t pts = case_data.hasKey("points") ? static_cast<size_t>(case_data["points"].as<long int>()) : 1;

            TestDomain1D domain(nv, pts);
            shared_ptr<vector<double>> shared_data = make_shared<vector<double>>(domain.size(), 0.0);
            domain.setData(shared_data);

            vector<TestDomain1D> multi_domains;

            if (case_data.hasKey("operations")) {
                const auto& ops = case_data["operations"].asVector<AnyMap>();
                for (const auto& op : ops) {
                    string op_name = op["op"].asString();

                    if (op_name == "query_basic") {
                        out_rec["points"] = static_cast<long int>(domain.nPoints());
                        out_rec["size"] = static_cast<long int>(domain.size());
                        out_rec["bandwidth"] = static_cast<long int>(domain.bandwidth());
                        out_rec["zmin"] = domain.zmin();
                        out_rec["zmax"] = domain.zmax();
                        out_rec["domain_type"] = domain.domainType();
                    } else if (op_name == "resize") {
                        size_t r_nv = static_cast<size_t>(op["nv"].as<long int>());
                        size_t r_pts = static_cast<size_t>(op["points"].as<long int>());
                        domain.resize(r_nv, r_pts);
                        shared_data = make_shared<vector<double>>(domain.size(), 0.0);
                        domain.setData(shared_data);
                    } else if (op_name == "query_tolerances") {
                        vector<double> s_rtol, s_atol, t_rtol, t_atol;
                        for (size_t n = 0; n < domain.nComponents(); n++) {
                            s_rtol.push_back(domain.steady_rtol(n));
                            s_atol.push_back(domain.steady_atol(n));
                            t_rtol.push_back(domain.transient_rtol(n));
                            t_atol.push_back(domain.transient_atol(n));
                        }
                        out_rec["steady_rtol"] = s_rtol;
                        out_rec["steady_atol"] = s_atol;
                        out_rec["transient_rtol"] = t_rtol;
                        out_rec["transient_atol"] = t_atol;
                    } else if (op_name == "set_component_names") {
                        const auto& names = op["names"].asVector<string>();
                        for (size_t i = 0; i < names.size() && i < domain.nComponents(); i++) {
                            domain.setComponentName(i, names[i]);
                        }
                    } else if (op_name == "query_component_names") {
                        vector<string> names;
                        for (size_t i = 0; i < domain.nComponents(); i++) {
                            names.push_back(domain.componentName(i));
                        }
                        out_rec["component_names"] = names;
                    } else if (op_name == "query_component_indices") {
                        const auto& query_names = op["names"].asVector<string>();
                        vector<long int> indices;
                        vector<bool> has_comps;
                        for (const auto& name : query_names) {
                            indices.push_back(static_cast<long int>(domain.componentIndex(name)));
                            has_comps.push_back(domain.hasComponent(name));
                        }
                        out_rec["queried_indices"] = indices;
                        out_rec["has_components"] = has_comps;
                    } else if (op_name == "set_bounds") {
                        size_t n = static_cast<size_t>(op["n"].as<long int>());
                        double lower = op["lower"].asDouble();
                        double upper = op["upper"].asDouble();
                        domain.setBounds(n, lower, upper);
                    } else if (op_name == "query_bounds") {
                        vector<double> lower, upper;
                        for (size_t n = 0; n < domain.nComponents(); n++) {
                            lower.push_back(domain.lowerBound(n));
                            upper.push_back(domain.upperBound(n));
                        }
                        out_rec["lower_bounds"] = lower;
                        out_rec["upper_bounds"] = upper;
                    } else if (op_name == "set_steady_tolerances") {
                        double rtol = op["rtol"].asDouble();
                        double atol = op["atol"].asDouble();
                        long int n = op["n"].as<long int>();
                        domain.setSteadyTolerances(rtol, atol, (n < 0) ? npos : static_cast<size_t>(n));
                    } else if (op_name == "set_transient_tolerances") {
                        double rtol = op["rtol"].asDouble();
                        double atol = op["atol"].asDouble();
                        long int n = op["n"].as<long int>();
                        domain.setTransientTolerances(rtol, atol, (n < 0) ? npos : static_cast<size_t>(n));
                    } else if (op_name == "init_time_integ") {
                        double dt = op["dt"].asDouble();
                        vector<double> x0(domain.size(), 1.0);
                        domain.initTimeInteg(dt, x0);
                    } else if (op_name == "init_time_integ_with_state") {
                        double dt = op["dt"].asDouble();
                        const auto& state = op["state"].asVector<double>();
                        domain.initTimeInteg(dt, state);
                    } else if (op_name == "set_steady_mode") {
                        domain.setSteadyMode();
                    } else if (op_name == "query_mode_and_tolerances") {
                        out_rec["steady_flag"] = domain.steady();
                        out_rec["transient_flag"] = domain.transient();
                        vector<double> cur_rtol, cur_atol;
                        for (size_t n = 0; n < domain.nComponents(); n++) {
                            cur_rtol.push_back(domain.rtol(n));
                            cur_atol.push_back(domain.atol(n));
                        }
                        out_rec["current_rtol"] = cur_rtol;
                        out_rec["current_atol"] = cur_atol;
                    } else if (op_name == "query_prev_soln") {
                        vector<double> prev;
                        for (size_t j = 0; j < domain.nPoints(); j++) {
                            for (size_t n = 0; n < domain.nComponents(); n++) {
                                prev.push_back(domain.prevSoln(n, j));
                            }
                        }
                        out_rec["prev_solution"] = prev;
                    } else if (op_name == "setup_grid") {
                        const auto& z = op["z"].asVector<double>();
                        domain.setupGrid(z);
                    } else if (op_name == "setup_uniform_grid") {
                        size_t unif_pts = static_cast<size_t>(op["points"].as<long int>());
                        double length = op["length"].asDouble();
                        double start = op["start"].asDouble();
                        domain.setupUniformGrid(unif_pts, length, start);
                    } else if (op_name == "query_grid") {
                        vector<double> z_pts;
                        for (size_t j = 0; j < domain.nPoints(); j++) {
                            z_pts.push_back(domain.z(j));
                        }
                        out_rec["grid_points"] = z_pts;
                        out_rec["zmin"] = domain.zmin();
                        out_rec["zmax"] = domain.zmax();
                    } else if (op_name == "reset_bad_values") {
                        auto state = op["state"].asVector<double>();
                        domain.resetBadValues(state);
                        out_rec["clamped_state"] = state;
                    } else if (op_name == "verify_topology_3") {
                        TestDomain1D d0(1, 1);
                        TestDomain1D d1(3, 5);
                        TestDomain1D d2(1, 1);
                        d0.append(&d1);
                        d1.append(&d2);

                        out_rec["d0_loc"] = static_cast<long int>(d0.loc());
                        out_rec["d0_firstPoint"] = static_cast<long int>(d0.firstPoint());
                        out_rec["d0_lastPoint"] = static_cast<long int>(d0.lastPoint());

                        out_rec["d1_loc"] = static_cast<long int>(d1.loc());
                        out_rec["d1_firstPoint"] = static_cast<long int>(d1.firstPoint());
                        out_rec["d1_lastPoint"] = static_cast<long int>(d1.lastPoint());

                        out_rec["d2_loc"] = static_cast<long int>(d2.loc());
                        out_rec["d2_firstPoint"] = static_cast<long int>(d2.firstPoint());
                        out_rec["d2_lastPoint"] = static_cast<long int>(d2.lastPoint());

                        out_rec["d1_global_idx_2_3"] = static_cast<long int>(d1.loc() + d1.index(2, 3));
                        out_rec["has_d0_left"] = (d0.left() != nullptr);
                        out_rec["has_d0_right"] = (d0.right() != nullptr);
                        out_rec["has_d2_right"] = (d2.right() != nullptr);
                    } else if (op_name == "verify_topology_5") {
                        const auto& mdoms = case_data["multi_domain"].asVector<AnyMap>();
                        vector<unique_ptr<TestDomain1D>> dom_ptrs;
                        for (const auto& md : mdoms) {
                            dom_ptrs.push_back(make_unique<TestDomain1D>(
                                md["nv"].as<long int>(),
                                md["points"].as<long int>()
                            ));
                        }
                        for (size_t k = 0; k + 1 < dom_ptrs.size(); k++) {
                            dom_ptrs[k]->append(dom_ptrs[k + 1].get());
                        }
                        vector<long int> locs, firsts, lasts;
                        for (const auto& dp : dom_ptrs) {
                            locs.push_back(static_cast<long int>(dp->loc()));
                            firsts.push_back(static_cast<long int>(dp->firstPoint()));
                            lasts.push_back(static_cast<long int>(dp->lastPoint()));
                        }
                        out_rec["topology5_locs"] = locs;
                        out_rec["topology5_firstPoints"] = firsts;
                        out_rec["topology5_lastPoints"] = lasts;
                    } else if (op_name == "set_refine_criteria") {
                        double ratio = op["ratio"].asDouble();
                        double slope = op["slope"].asDouble();
                        double curve = op["curve"].asDouble();
                        double prune = op["prune"].asDouble();
                        domain.setRefineCriteria(ratio, slope, curve, prune);
                    } else if (op_name == "query_refine_criteria") {
                        out_rec["refine_criteria"] = domain.getRefineCriteria();
                    } else if (op_name == "set_jacobian_mode") {
                        domain.setJacobianMode(op["mode"].asString());
                    } else if (op_name == "query_jacobian_mode") {
                        out_rec["jacobian_mode"] = domain.jacobianMode();
                    } else if (op_name == "set_invalid_jacobian_mode") {
                        bool caught = false;
                        try {
                            domain.setJacobianMode(op["mode"].asString());
                        } catch (const CanteraError&) {
                            caught = true;
                        }
                        out_rec["caught_invalid_jacobian_mode"] = caught;
                    } else if (op_name == "query_meta") {
                        AnyMap meta = domain.getMeta();
                        out_rec["meta_yaml"] = meta.toYamlString();
                    } else if (op_name == "test_element_indexing_and_access") {
                        vector<double> x_arr(domain.size(), 0.0);
                        vector<long int> indices;
                        for (size_t j = 0; j < domain.nPoints(); j++) {
                            for (size_t n = 0; n < domain.nComponents(); n++) {
                                size_t idx = domain.index(n, j);
                                indices.push_back(static_cast<long int>(idx));
                                x_arr[idx] = 10.0 * (n + 1) + 0.1 * (j + 1);
                            }
                        }
                        vector<double> queried_values;
                        for (size_t j = 0; j < domain.nPoints(); j++) {
                            for (size_t n = 0; n < domain.nComponents(); n++) {
                                queried_values.push_back(x_arr[domain.index(n, j)]);
                            }
                        }
                        out_rec["element_indices"] = indices;
                        out_rec["queried_values"] = queried_values;
                    } else if (op_name == "test_initial_solution") {
                        vector<double> init_soln(domain.size(), 0.0);
                        domain._getInitialSoln(init_soln);
                        out_rec["initial_solution"] = init_soln;
                    } else if (op_name == "set_flat_profile") {
                        size_t comp = static_cast<size_t>(op["component"].as<long int>());
                        double val = op["value"].asDouble();
                        for (size_t j = 0; j < domain.nPoints(); j++) {
                            (*shared_data)[domain.index(comp, j)] = val;
                        }
                    } else if (op_name == "query_state") {
                        out_rec["shared_state"] = *shared_data;
                    }
                }
            }
        } catch (const CanteraError& err) {
            out_rec["success"] = false;
            out_rec["exception_thrown"] = true;
            out_rec["error_message"] = err.what();
        } catch (const exception& ex) {
            out_rec["success"] = false;
            out_rec["exception_thrown"] = true;
            out_rec["error_message"] = ex.what();
        }

        results.push_back(out_rec);
        case_idx++;
    }

    AnyMap final_out;
    final_out["cases"] = results;

    ofstream outfile(output_path);
    outfile << final_out.toYamlString();
    outfile.close();

    cout << "Executed " << results.size() << " native Domain1D cases -> " << output_path << endl;
    return 0;
}
