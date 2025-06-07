import subprocess

subprocess.run([
    "python", "train.py",
    "--data_root", "./aligned_data",
    "--train_file", "./train_list.txt",
    "--backbone_type", "ResNet",
    "--backbone_conf_file", "../backbone_conf.yaml",
    "--head_type", "ArcFace",
    "--head_conf_file", "../head_conf.yaml",
    "--lr", "0.01",
    "--out_dir", "./results/mobileface_magface",
    "--epoches", "80",
    "--step", "30,50,70",
    "--print_freq", "200",
    "--save_freq", "5000",
    "--batch_size", "64",
    "--momentum", "0.9",
    "--log_dir", "../../log",
    "--tensorboardx_logdir", "mv-hrnet",
])
