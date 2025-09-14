# Additional Readme to get things going...

## Getting started:
1. Installation
```bash
pip3 install -r requirements_new.txt
```

2. Running: (in the `kb_new` folder.)
```bash
python3 run_reference_timings.py
```

## Generally useful command:
```bash
export HF_HOME="/workspace"
```

```bash
ipython kernel install --user --name=kernel_bench
```

Press: `CMD+Shift+P` to select python interpreter.

```bash
git config --global user.name "Nikolai Rozanov"
git config --global user.email "nikolai.rozanov@gmail.com"
```

----
## Concrete next steps:
1. Create a static Correctness dataset (inputs & outputs) x num_trials (e.g. 5)
2. Create a basic prompt pipeline 1. (code => output_code) 2. Evaluation of output_code (based on Correctness dataset & timing json)
3. Enhance the above using Test time search & better compilation feedback etc.