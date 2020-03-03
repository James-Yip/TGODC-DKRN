import os


def create_logs(logs_path):
    dir_path = os.path.dirname(logs_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    logs = open(logs_path, "a+")
    logs.close()


def add_logs(logs_path, logs_contents,):
    logs = open(logs_path, "a")
    for logs_content in logs_contents:
        print(logs_content)
        logs.write(logs_content)
        logs.write('\n')
    logs.close()


def add_log(logs_path, logs_content):
    logs = open(logs_path, "a")
    print(logs_content)
    logs.write(logs_content)
    logs.write('\n')
    logs.close()