import os


def create_log_folder():
    path = "../result/"
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, "log")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def write_line(line, name):
    folder = create_log_folder()
    path = os.path.join(folder, "{}.csv".format(name))
    with open(path, "a+") as fp:
        fp.write(line + "\n")


class ChromosomeLogger:
    def __init__(self):
        self.logs = []
        self.current_chromosome = None

    def log_chromosome(self, chromosome):
        self.logs.append({"chromosome": chromosome, "data": []})

    def log_epoch(self, epoch_data):
        self.logs[-1]["data"].append(epoch_data)
