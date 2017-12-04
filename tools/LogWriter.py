from tensorboardX import SummaryWriter

import pdb

"""
Ref: https://github.com/lanpa/tensorboard-pytorch
"""
class LogWriter(SummaryWriter):
    def __init__(self, log_dir):
        #pdb.set_trace()
        super(LogWriter, self).__init__(str(log_dir))
        self.log_dir = self.file_writer.get_logdir()
