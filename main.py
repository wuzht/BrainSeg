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