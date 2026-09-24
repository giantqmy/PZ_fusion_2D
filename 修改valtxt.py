root = "/media/ddc/新加卷/hys/qmy/pz/images/"

with open(r"/media/ddc/新加卷/hys/qmy/pz/val.txt", "r") as f:
    lines = f.readlines()

with open(r"/media/ddc/新加卷/hys/qmy/pz/val_abs.txt", "w") as f:
    for line in lines:
        name = line.strip().split("/")[-1]
        f.write(root + name + "\n")
