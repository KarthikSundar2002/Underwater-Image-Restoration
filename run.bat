
cd C:\\Users\\givew\\PycharmProjects\\Underwater-Image-Restoration

.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf mix --optim adamw
.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf fflmix --optim adamw
.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf L1withColor --optim adamw
.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf charbonnier --optim adamw
.venv\Scripts\python.exe main.py -a SpectralTransformer --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf perceptual --optim adamw

.venv\Scripts\python.exe main.py -a NewBigModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf mix --optim adamw
.venv\Scripts\python.exe main.py -a NewBigModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf fflmix --optim adamw
.venv\Scripts\python.exe main.py -a NewBigModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf L1withColor --optim adamw
.venv\Scripts\python.exe main.py -a NewBigModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf charbonnier --optim adamw
.venv\Scripts\python.exe main.py -a NewBigModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf perceptual --optim adamw

.venv\Scripts\python.exe main.py -a NewModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf mix --optim adamw
.venv\Scripts\python.exe main.py -a NewModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf fflmix --optim adamw
.venv\Scripts\python.exe main.py -a NewModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf L1withColor --optim adamw
.venv\Scripts\python.exe main.py -a NewModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf charbonnier --optim adamw
.venv\Scripts\python.exe main.py -a NewModel --lr 0.0002 --max-epoch 10 --train-batch-size 4 --test-batch-size 4 -lossf perceptual --optim adamw
pause