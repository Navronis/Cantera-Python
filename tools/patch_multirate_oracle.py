from pathlib import Path

fpath = Path("outputs/cantera-python/tools/run_native_kinetics_family5_oracle.cpp")
text = fpath.read_text(encoding="utf-8")

old_block = """                auto rate = newReactionRate(rtype);
                string actualType = rate->type();
                auto multi = rate->newMultiRate();

                bool caughtEmpty = false;
                try {
                    multi->type();
                } catch (const CanteraError&) {
                    caughtEmpty = true;
                }

                multi->add(0, *rate);
                string multiType = multi->type();

                // Test replacing with identical type
                auto rate2 = newReactionRate(rtype);
                bool repOk = multi->replace(0, *rate2);"""

new_block = """                shared_ptr<ReactionRate> rate;
                if (rtype == "constant-vibrational-relaxation") {
                    rate = make_shared<ConstantVibrationalRelaxationRate>(1.0e6);
                } else {
                    rate = newReactionRate(rtype);
                }
                string actualType = rate->type();
                auto multi = rate->newMultiRate();

                bool caughtEmpty = false;
                try {
                    multi->type();
                } catch (const CanteraError&) {
                    caughtEmpty = true;
                }

                multi->add(0, *rate);
                string multiType = multi->type();

                // Test replacing with identical type
                shared_ptr<ReactionRate> rate2;
                if (rtype == "constant-vibrational-relaxation") {
                    rate2 = make_shared<ConstantVibrationalRelaxationRate>(2.0e6);
                } else {
                    rate2 = newReactionRate(rtype);
                }
                bool repOk = multi->replace(0, *rate2);"""

assert old_block in text, "old_block not found in oracle.cpp"
text = text.replace(old_block, new_block, 1)
fpath.write_text(text, encoding="utf-8")
print("SUCCESS: updated multirate handling in oracle.cpp")
