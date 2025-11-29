import os
import matplotlib.pyplot as plt

train_path = "dataset/train"

classes = sorted(os.listdir(train_path))
counts = []

for cls in classes:
    cls_path = os.path.join(train_path, cls)
    counts.append(len(os.listdir(cls_path)))

print("Classes:", classes)
print("Counts:", counts)

plt.figure()
plt.bar(classes, counts)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Training Images per Class")
plt.savefig("class_distribution.png")
print("Saved class_distribution.png")
