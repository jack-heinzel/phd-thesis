import os

os.chdir("data_release")
for subdir in ["AllCBC", "BGP", "NS", "BBH"]:
    os.system(f"tar -cvf analyses_{subdir}.tar *{subdir}*h5")
    os.system(f"rm *{subdir}*h5")

os.chdir("../figures")
os.system("tar -cvf figures.tar *pdf")

os.chdir("../figure_scripts")
os.system("tar -cvf figure_scripts.tar *py")

os.chdir("../o4a_event_list")
os.system("tar -cvf o4a_event_list.tar *txt")
