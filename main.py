from config import get_cfg
from op import Operation

cfg = get_cfg()

operation = Operation(cfg)

operation.fit()