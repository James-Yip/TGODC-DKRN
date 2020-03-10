import os


def create_logs(logs_path):
    dir_path = os.path.dirname(logs_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    logs = open(logs_path, "a+")
    logs.close()


def add_logs(logs_path, logs_contents, print_details=True):
    logs = open(logs_path, "a")
    for logs_content in logs_contents:
        if print_details:
            print(logs_content)
        logs.write(logs_content + '\n')
    logs.close()


def add_log(logs_path, logs_content, print_details=True):
    logs = open(logs_path, "a")
    if print_details:
        print(logs_content)
    logs.write(logs_content + '\n')
    logs.close()
