from tools import choose_gpu
import time

# while True:
#     gpu_id, memory_gpu = choose_gpu()
    
#     print(memory_gpu, max(memory_gpu))

#     if max(memory_gpu) >= 6600:
#         break
#     time.sleep(4)


from config import get_cfg
from op import Operation

cfg = get_cfg()

operation = Operation(cfg)

operation.fit()

# operation.load(path='./exp/D-All-brains18=0409-151829/model_best.pt')

# # 用loader评价模型，不采样
# dices, loss = operation.eval_model(operation.val_loader)
# print(dices[1:].mean())
# dices, loss = operation.eval_model(operation.train_loader)
# print(dices[1:].mean())

# # 用data评价模型，不采样
# dice1, dice2 = operation.eval_model_dices(operation.val_data)
# dice1, dice2 = operation.eval_model_dices(operation.train_data)

# # 评价val，采样
# # operation.cfg.sample_T = 30
# dice1, dice2 = operation.eval_sample_model_dices(operation.val_data, True)

# # 评价train，采样
# dice1, dice2 = operation.eval_sample_model_dices(operation.train_data, True)
