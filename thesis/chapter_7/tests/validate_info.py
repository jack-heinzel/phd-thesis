import yaml


def load_info():
    """ Load the info file: this assumes it is run from the top level """
    info_file = "info.txt"
    with open(info_file, "r") as file:
        info = yaml.safe_load(file)
    print("Successfully loaded info.txt")
    return info


def validate_keys(info):
    required_keys = ["title", "working_title", "DCCID", "status", "group", "target_journal"]
    missing_keys = []
    for key in required_keys:
        if key not in info:
            missing_keys.append(key)

    if len(missing_keys) > 0:
        raise ValueError(f"info.txt is missing the keys: {missing_keys}")
    else:
        print("Successfully validated the required keys in info.txt")


if __name__ == "__main__":
    info = load_info()
    validate_keys(info)
