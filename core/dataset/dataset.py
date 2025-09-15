from torch.utils.data.dataset import Dataset


class MAMLDataset(Dataset):

    def __init__(self, data_path, batch_size, n_way=5, k_shot=1, q_query=1):

        self.file_list = self.get_file_list(data_path)
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

    def get_file_list(self, data_path):
        raise NotImplementedError('get_file_list function not implemented!')

    def get_one_task_data(self):
        raise NotImplementedError('get_one_task_data function not implemented!')

    def __len__(self):
        # return len(self.file_list)*11 // self.batch_size       ### file_list*110是因为总共10种类别调制信号 原来每种调制信号内又有11个不同信噪比组成的11个文件夹，所以为110个filelist
        #                                                             # 但现将信噪比文件夹去除了 因为只需对调制信号种类进行分类 并不需要对信噪比大小进行识别分类

        return len(self.file_list)* 200// self.batch_size  ### file_list*11是因为总共10种类别调制信号 原来每种调制信号内又有11个不同信噪比组成的11个文件夹，所以为110个filelist

        # return len(self.file_list)*1600  // self.batch_size       ## Omniglot文件夹有字母种类也有字母种类下的数字种类，len(self.file_list)就是（语言种类*字母数字）

    def __getitem__(self, index):
        return self.get_one_task_data()



class MAMLDataset0(Dataset):

    def __init__(self, img_data_path, seq_data_path, batch_size, n_way=5, k_shot=1, q_query=1):

        self.file_list1 = self.get_file_list(img_data_path)
        self.file_list2 = self.get_file_list(seq_data_path)
        self.batch_size = batch_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

    def get_file_list(self, data_path):
        raise NotImplementedError('get_file_list function not implemented!')

    def get_one_task_data(self):
        raise NotImplementedError('get_one_task_data function not implemented!')

    def __len__(self):

        return len(self.file_list1) * 200 // self.batch_size

    def __getitem__(self, index):
        return self.get_one_task_data()



