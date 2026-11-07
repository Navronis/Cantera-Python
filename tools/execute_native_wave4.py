import subprocess
import time

def run():
    print("Compiling tools/run_native_wave4.cpp in WSL...")
    t0 = time.time()
    compile_cmd = (
        "wsl g++ -std=c++20 -O3 tools/run_native_wave4.cpp "
        "-I/home/nebula/cantera-pinned-oracle/include "
        "-L/home/nebula/cantera-pinned-oracle/build/lib "
        "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib "
        "-lcantera_shared -o tools/run_native_wave4"
    )
    subprocess.run(compile_cmd, shell=True, check=True)
    t_comp = time.time() - t0
    print(f"Compilation succeeded in {t_comp:.2f}s. Running native evaluation across 10,000 cases...")

    t1 = time.time()
    run_cmd = "wsl ./tools/run_native_wave4 audit/corpus_wave4.jsonl audit/native_wave4_results.jsonl"
    subprocess.run(run_cmd, shell=True, check=True)
    t_run = time.time() - t1
    print(f"Native evaluation completed in {t_run:.2f}s!")

if __name__ == "__main__":
    run()
