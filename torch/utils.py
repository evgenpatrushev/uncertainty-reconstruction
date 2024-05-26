import os


class IOStream:
    def __init__(self, path, do_valid=False, is_training=True):

        self.is_training = is_training
        if is_training:
            self.init_training(path, do_valid)

    def init_training(self, path, do_valid):
        train_summary_path = path + "/train/"
        if not os.path.exists(train_summary_path):
            os.makedirs(train_summary_path)
        self.do_valid = do_valid
        if self.do_valid:
            valid_summary_path = path + "/valid/"
            if not os.path.exists(valid_summary_path):
                os.makedirs(valid_summary_path)
        self.text_log = open(path + '/run.log', 'a')

    def write_to_terminal(self, text):

        if self.is_training:
            print(text)
            self.text_log.write(text + '\n')
            self.text_log.flush()
        else:
            print(text)

    def close(self):
        if self.is_training:
            self.text_log.close()
