import os.path

from imagecorruptions import corrupt


class DataConfig:
    data_name = ""
    root_dir_A = ""
    root_dir_B = ""
    label_transform = ""
    data_root_path = ""
    def get_data_config(self, data_name, data_root):
        self.data_name = data_name
        severity = 5
        self.label_transform = "norm"
        if 'HH' in self.data_name:
            self.root_dir_A = os.path.join(data_root, data_name)
            self.root_dir_B = os.path.join(data_root, data_name)
        else:
            dataset_felix = ''
            if '-' in self.data_name[:-4]:
                dataset_felix, string = self.data_name[:-4].split("-", 1)
                if '-' in string:
                    A_type, B_type = string.split('-', 1)
                else:
                    A_type, B_type = string, string
                if A_type =='256':
                    self.root_dir_A = os.path.join(f'{data_root}/{dataset_felix}', f"{dataset_felix}-256")
                else:
                    self.data_root_path = f'{data_root}/{dataset_felix}/severity{severity}'
                    self.root_dir_A = os.path.join(self.data_root_path, f"{dataset_felix}-{A_type}-256")
                if B_type == '256':
                    self.root_dir_B = os.path.join(f'{data_root}/{dataset_felix}', f"{dataset_felix}-256")
                else:
                    self.data_root_path = f'{data_root}/{dataset_felix}/severity{severity}'
                    self.root_dir_B = os.path.join(self.data_root_path, f"{dataset_felix}-{B_type}-256")
            else:
                dataset_felix = self.data_name[:-4]
                self.root_dir_A = f'{data_root}/{dataset_felix}/{self.data_name}'
                self.root_dir_B = f'{data_root}/{dataset_felix}/{self.data_name}'
            print(f"Pre image comes from {self.root_dir_A}")
            print(f"Post image comes from {self.root_dir_B}")
            dataset_list = ['LEVIR', 'EGY', 'WHU', 'DSIFN', 'SYSU', 'CLCD']
            if dataset_felix not in dataset_list:
                raise TypeError('%s has not defined' % dataset_felix)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir_A)
    print(data.label_transform)

