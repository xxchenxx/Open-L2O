# Environment setup

```
conda create --name tf1.15  python=3.6
conda activate tf1.15
pip install --user nvidia-pyindex
pip install --user nvidia-tensorflow[horovod]
pip install mock
pip install dm-sonnet==1.11
pip install dill
pip install tensorflow_probability==0.7.0
```

# Run the following commands to reproduce the results in the paper
python train_dm.py --save_path=test_1 --problem=lasso-fixed-10-5 --if_cl=True --if_mt=True
python evaluate_dm.py --path=test/cw.l2l-0 --num_steps=10000 --problem=lasso-fixed-10-5-test --output_path=test_output --seed=2

python train_dm.py --save_path=test_2 --problem=lasso-fixed-50-25 --if_cl=True --if_mt=True
python evaluate_dm.py --path=test/cw.l2l-0 --num_steps=10000 --problem=lasso-fixed-50-25-test --output_path=test_output --seed=2

python train_dm.py --save_path=test_3 --problem=rastrigin-fixed-2 --if_cl=True --if_mt=True
python evaluate_dm.py --path=test/cw.l2l-0 --num_steps=10000 --problem=rastrigin-fixed-2-test --output_path=test_output --seed=2



