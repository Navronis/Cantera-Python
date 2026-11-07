with open('outputs/cantera-python/tools/run_native_boundary1d_oracle.cpp', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace surf_mech definition
text = text.replace(
    'string surf_mech = "/home/nebula/cantera-pinned-oracle/test/data/simple_surface.yaml";',
    'string surf_mech = "ptcombust.yaml";'
)

# Case 15: replace "surf" with "Pt_surf"
old_case15 = """        else if (case_id == "boundary1d_15_reactingsurf1d_properties") {
            auto sol_surf = newSolution(surf_mech, "surf");"""
new_case15 = """        else if (case_id == "boundary1d_15_reactingsurf1d_properties") {
            auto sol_surf = newSolution(surf_mech, "Pt_surf");"""
assert old_case15 in text, "old_case15 not found"
text = text.replace(old_case15, new_case15)

# Case 16: replace "surf" with "Pt_surf" and size 11
old_case16 = """        else if (case_id == "boundary1d_16_reactingsurf1d_bounds_resetbadvalues") {
            auto sol_surf = newSolution(surf_mech, "surf");
            TestReactingSurf1D rsurf(sol_surf, "test_rsurf");
            rsurf.testInit();

            auto surf_phase = dynamic_pointer_cast<SurfPhase>(sol_surf->thermo());
            size_t nsp = surf_phase->nSpecies();

            out << "        \\"lower_bounds\\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rsurf.lowerBound(k);
            }
            out << "],\\n";
            out << "        \\"upper_bounds\\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rsurf.upperBound(k);
            }
            out << "],\\n";

            vector<double> cov = {0.5, 0.5, 0.5, 0.5, 0.5};"""
new_case16 = """        else if (case_id == "boundary1d_16_reactingsurf1d_bounds_resetbadvalues") {
            auto sol_surf = newSolution(surf_mech, "Pt_surf");
            TestReactingSurf1D rsurf(sol_surf, "test_rsurf");
            rsurf.testInit();

            auto surf_phase = dynamic_pointer_cast<SurfPhase>(sol_surf->thermo());
            size_t nsp = surf_phase->nSpecies();

            out << "        \\"lower_bounds\\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rsurf.lowerBound(k);
            }
            out << "],\\n";
            out << "        \\"upper_bounds\\": [";
            for (size_t k = 0; k < nsp; k++) {
                if (k > 0) out << ", ";
                out << setprecision(16) << rsurf.upperBound(k);
            }
            out << "],\\n";

            vector<double> cov(nsp, 0.5);"""
assert old_case16 in text, "old_case16 not found"
text = text.replace(old_case16, new_case16)

# Case 17:
old_case17 = """        else if (case_id == "boundary1d_17_reactingsurf1d_eval_steady") {
            auto sol_gas = newSolution(surf_mech, "gas");
            auto sol_surf = newSolution(surf_mech, "surf", "none", {sol_gas});
            auto flow = newFlow1D("gas-flow", sol_gas);
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(rsurf);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            rsurf->setTemperature(900.0);
            container.init();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t rs_loc = rsurf->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_T] = 880.0;
                xg[pt + c_offset_Y] = 0.25;
                xg[pt + c_offset_Y + 1] = 0.25;
                xg[pt + c_offset_Y + 2] = 0.25;
                xg[pt + c_offset_Y + 3] = 0.25;
            }

            vector<double> cov = {0.1, 0.2, 0.05, 0.15, 0.5};"""
new_case17 = """        else if (case_id == "boundary1d_17_reactingsurf1d_eval_steady") {
            auto sol_gas = newSolution(surf_mech, "gas");
            auto sol_surf = newSolution(surf_mech, "Pt_surf", "none", {sol_gas});
            auto flow = newFlow1D("gas-flow", sol_gas);
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(rsurf);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            rsurf->setTemperature(900.0);
            container.init();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t rs_loc = rsurf->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            size_t nsp_gas = sol_gas->thermo()->nSpecies();

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_T] = 880.0;
                for (size_t k = 0; k < nsp_gas; k++) {
                    xg[pt + c_offset_Y + k] = 1.0 / nsp_gas;
                }
            }

            vector<double> cov = {0.30, 0.10, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10};"""
assert old_case17 in text, "old_case17 not found"
text = text.replace(old_case17, new_case17)

# Case 18:
old_case18 = """        else if (case_id == "boundary1d_18_reactingsurf1d_eval_transient") {
            auto sol_gas = newSolution(surf_mech, "gas");
            auto sol_surf = newSolution(surf_mech, "surf", "none", {sol_gas});
            auto flow = newFlow1D("gas-flow", sol_gas);
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(rsurf);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            rsurf->setTemperature(900.0);
            container.init();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t rs_loc = rsurf->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_T] = 880.0;
                xg[pt + c_offset_Y] = 0.25;
                xg[pt + c_offset_Y + 1] = 0.25;
                xg[pt + c_offset_Y + 2] = 0.25;
                xg[pt + c_offset_Y + 3] = 0.25;
            }

            vector<double> cov = {0.12, 0.18, 0.06, 0.14, 0.50};
            for (size_t k = 0; k < cov.size(); k++) {
                xg[rs_loc + k] = cov[k];
            }

            // Set previous solution
            vector<double> prev_cov = {0.10, 0.20, 0.05, 0.15, 0.50};
            vector<double> prev_x(container.size(), 0.0);
            for (size_t k = 0; k < prev_cov.size(); k++) {
                prev_x[rs_loc + k] = prev_cov[k];
            }
            rsurf->initTimeInteg(1.0/1000.0, span<const double>(prev_x.data() + rs_loc, 5));"""
new_case18 = """        else if (case_id == "boundary1d_18_reactingsurf1d_eval_transient") {
            auto sol_gas = newSolution(surf_mech, "gas");
            auto sol_surf = newSolution(surf_mech, "Pt_surf", "none", {sol_gas});
            auto flow = newFlow1D("gas-flow", sol_gas);
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(rsurf);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            rsurf->setTemperature(900.0);
            container.init();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t rs_loc = rsurf->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            size_t nsp_gas = sol_gas->thermo()->nSpecies();

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_T] = 880.0;
                for (size_t k = 0; k < nsp_gas; k++) {
                    xg[pt + c_offset_Y + k] = 1.0 / nsp_gas;
                }
            }

            vector<double> cov = {0.32, 0.10, 0.05, 0.05, 0.08, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10};
            for (size_t k = 0; k < cov.size(); k++) {
                xg[rs_loc + k] = cov[k];
            }

            // Set previous solution
            vector<double> prev_cov = {0.30, 0.10, 0.05, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10};
            vector<double> prev_x(container.size(), 0.0);
            for (size_t k = 0; k < prev_cov.size(); k++) {
                prev_x[rs_loc + k] = prev_cov[k];
            }
            rsurf->initTimeInteg(1.0/1000.0, span<const double>(prev_x.data() + rs_loc, 11));"""
assert old_case18 in text, "old_case18 not found"
text = text.replace(old_case18, new_case18)

# Case 19:
old_case19 = """        else if (case_id == "boundary1d_19_reactingsurf1d_eval_fixed_coverage") {
            auto sol_gas = newSolution(surf_mech, "gas");
            auto sol_surf = newSolution(surf_mech, "surf", "none", {sol_gas});
            auto flow = newFlow1D("gas-flow", sol_gas);
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(rsurf);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            rsurf->setTemperature(850.0);
            rsurf->enableCoverageEquations(false);
            container.init();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t rs_loc = rsurf->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_T] = 850.0;
                xg[pt + c_offset_Y] = 0.25;
                xg[pt + c_offset_Y + 1] = 0.25;
                xg[pt + c_offset_Y + 2] = 0.25;
                xg[pt + c_offset_Y + 3] = 0.25;
            }
            vector<double> cov = {0.15, 0.25, 0.05, 0.15, 0.40};"""
new_case19 = """        else if (case_id == "boundary1d_19_reactingsurf1d_eval_fixed_coverage") {
            auto sol_gas = newSolution(surf_mech, "gas");
            auto sol_surf = newSolution(surf_mech, "Pt_surf", "none", {sol_gas});
            auto flow = newFlow1D("gas-flow", sol_gas);
            auto rsurf = newDomain<ReactingSurf1D>("reacting-surface", sol_surf);
            OneDim container;
            container.addDomain(flow);
            container.addDomain(rsurf);
            vector<double> z = {0.0, 0.01};
            flow->setupGrid(z);
            container.resize();
            rsurf->setTemperature(850.0);
            rsurf->enableCoverageEquations(false);
            container.init();

            vector<double> xg(container.size(), 0.0);
            size_t fl_loc = flow->loc();
            size_t rs_loc = rsurf->loc();
            size_t last_j = flow->nPoints() - 1;
            size_t last_pt = fl_loc + flow->index(0, last_j);
            size_t nsp_gas = sol_gas->thermo()->nSpecies();

            for (size_t j = 0; j < flow->nPoints(); j++) {
                size_t pt = fl_loc + flow->index(0, j);
                xg[pt + c_offset_T] = 850.0;
                for (size_t k = 0; k < nsp_gas; k++) {
                    xg[pt + c_offset_Y + k] = 1.0 / nsp_gas;
                }
            }
            vector<double> cov = {0.35, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10};"""
assert old_case19 in text, "old_case19 not found"
text = text.replace(old_case19, new_case19)

with open('outputs/cantera-python/tools/run_native_boundary1d_oracle.cpp', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully updated oracle for ptcombust")
