
cd C:\\Users\\spatr\\PycharmProjects\\Underwater-Image-Restoration


.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.00002 --max-epoch 1000 --train-batch-size 1 --test-batch-size 1 -lossf L1 --optim adamw
.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.00002 --max-epoch 1000 --train-batch-size 1 --test-batch-size 1 -lossf mix --optim adamw

.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.0002 --max-epoch 1000 --train-batch-size 1 --test-batch-size 1 -lossf L1 --optim adamw
.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.0002 --max-epoch 1000 --train-batch-size 1 --test-batch-size 1 -lossf mix --optim adamw

.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.00001 --max-epoch 1000 --train-batch-size 1 --test-batch-size 1 -lossf L1 --optim adamw
.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.00001 --max-epoch 1000 --train-batch-size 1 --test-batch-size 1 -lossf mix --optim adamw

.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.000005 --max-epoch 1000 --train-batch-size 1 --test-batch-size 1 -lossf L1 --optim adamw
.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.000005 --max-epoch 1000 --train-batch-size 1 --test-batch-size 1 -lossf mix --optim adamw
pause